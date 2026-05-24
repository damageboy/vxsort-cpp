use std::{
    collections::{BTreeMap, HashSet},
    sync::{Arc, Mutex, mpsc},
    thread,
};

use gadget_synth::{
    Arch as SynthArch, DType as SynthDType, GadgetGraph, GadgetSynthesizer, PermutationGadget,
    SynthesisError, SynthesisOptions, VectorState,
};

use crate::bitonic_sorter::{BitonicSorter, BitonicStage};
use crate::runtime::{NullRuntimeSession, RuntimeEvent, RuntimeSession, StageProgressSnapshot};
use crate::scoring::{AssignedPath, AssignedPathKey, DummyScorer, PathCost, Scorer};
use crate::transition_table::{CompletePath, StateTuple, TransitionTable};
use crate::{ArchArg, DTypeArg};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WaveConfig {
    pub num_vecs: usize,
    pub arch: ArchArg,
    pub dtype: DTypeArg,
    pub gadget_depth: u8,
    pub natural_order: bool,
    pub retroactive_input: bool,
    pub top_k: Option<usize>,
    pub worker_count: usize,
    pub max_unique_outputs: usize,
}

pub struct WaveEngine {
    config: WaveConfig,
    scorer: Box<dyn Scorer>,
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
    wave_count: usize,
    exhausted_stages: HashSet<usize>,
    stalled_stages: HashSet<usize>,
    scored_path_keys: HashSet<AssignedPathKey>,
    scored_paths: Vec<ScoredPath>,
    next_scored_path_order: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredPath {
    path: CompletePath,
    assigned_path: AssignedPath,
    cost: PathCost,
    discovery_order: usize,
}

impl ScoredPath {
    pub fn new(
        path: CompletePath,
        assigned_path: AssignedPath,
        cost: PathCost,
        discovery_order: usize,
    ) -> Self {
        Self {
            path,
            assigned_path,
            cost,
            discovery_order,
        }
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
            path: self.path.clone(),
            assigned_path: self.assigned_path.clone(),
            cost,
            discovery_order: self.discovery_order,
        }
    }
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SynthesisJob {
    stage: usize,
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

    pub fn candidate_index(&self) -> usize {
        self.candidate_index
    }
}

#[derive(Clone, Debug, PartialEq)]
struct WorkerSynthesisJob {
    index: usize,
    job: SynthesisJob,
    graph: GadgetGraph,
    target_pairs: Vec<(u64, u64)>,
    stage_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct SynthesisJobOutput {
    job: SynthesisJob,
    results: Vec<(PermutationGadget, VectorState)>,
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageRunResult {
    stage: usize,
    attempts: usize,
    valid_outputs: usize,
    new_outputs: usize,
    new_transitions: usize,
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
}

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

impl WaveEngine {
    pub fn new(config: WaveConfig) -> Result<Self, SynthesisError> {
        Self::with_scorer(config, Box::new(DummyScorer))
    }

    pub fn with_scorer(
        config: WaveConfig,
        scorer: Box<dyn Scorer>,
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
        let mut transition_table = TransitionTable::new(stages.len());
        let mut exhausted_stages = HashSet::new();
        if config.retroactive_input {
            prepopulate_retroactive_stage0(&mut transition_table, &stages[0]);
            let forwarded = transition_table
                .get_unforwarded_outputs(0)
                .into_iter()
                .map(|(state_tuple, _)| state_tuple)
                .collect::<Vec<_>>();
            transition_table.mark_forwarded(0, &forwarded);
            exhausted_stages.insert(0);
        }

        let stage_count = stages.len();

        Ok(Self {
            config,
            scorer,
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
            wave_count: 0,
            exhausted_stages,
            stalled_stages: HashSet::new(),
            scored_path_keys: HashSet::new(),
            scored_paths: Vec::new(),
            next_scored_path_order: 0,
        })
    }

    pub fn config(&self) -> WaveConfig {
        self.config
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

    pub fn best_score(&self) -> Option<f64> {
        self.scored_paths
            .iter()
            .map(|path| path.cost().score())
            .min_by(f64::total_cmp)
    }

    pub fn discover_paths_for_transition(
        &mut self,
        stage: usize,
        input_tuple: &StateTuple,
        output_tuple: &StateTuple,
        max_paths: Option<usize>,
    ) -> usize {
        let paths = self.transition_table.trace_paths_ending_with(
            stage,
            input_tuple,
            output_tuple,
            max_paths,
            None,
        );
        let mut discovered = 0;
        for path in paths {
            let limit = self.config.top_k.unwrap_or(1);
            for assigned_path in
                self.scorer
                    .assign_path_gadgets_k_best(&path, &self.transition_table, limit)
            {
                let key = assigned_path.selection_key();
                if self.scored_path_keys.contains(&key) {
                    continue;
                }
                let cost = self.scorer.score_assigned_path(&assigned_path);
                self.scored_path_keys.insert(key);
                self.scored_paths.push(ScoredPath::new(
                    path.clone(),
                    assigned_path,
                    cost,
                    self.next_scored_path_order,
                ));
                self.next_scored_path_order += 1;
                self.retain_top_k_paths();
                discovered += 1;
            }
        }
        discovered
    }

    pub fn record_transition(
        &mut self,
        stage: usize,
        input_state: &VectorState,
        output_state: &VectorState,
        gadget: PermutationGadget,
    ) -> TransitionRecordResult {
        let input_tuple = input_state.as_tuple();
        let output_tuple = output_state.as_tuple();
        let transition_added =
            self.transition_table
                .add_transition(stage, input_state, output_state, gadget);
        let discovered_paths = if transition_added && stage + 1 == self.stages.len() {
            self.discover_paths_for_transition(stage, &input_tuple, &output_tuple, None)
        } else {
            0
        };

        TransitionRecordResult {
            transition_added,
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
            self.ordered_inputs_for_stage(stage, input_states.to_vec())
        } else if stage == 0 {
            vec![self.initial_state.clone()]
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
                for input_state in &inputs {
                    let input_tuple = input_state.as_tuple();
                    if self
                        .transition_table
                        .was_attempted(stage, &input_tuple, candidate_index)
                    {
                        continue;
                    }

                    jobs.push(SynthesisJob {
                        stage,
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

        Ok(self.apply_synthesis_output(output))
    }

    fn apply_synthesis_output(&mut self, output: SynthesisJobOutput) -> JobExecutionResult {
        let job = output.job;

        self.transition_table.record_attempt(job.stage, 1);
        self.transition_table.record_attempted_pair(
            job.stage,
            &job.input_state.as_tuple(),
            job.candidate_index,
        );

        let valid_output_count = output.results.len();
        let mut new_transition_count = 0;
        let mut discovered_paths = 0;
        for (gadget, output_state) in output.results {
            let result = self.record_transition(job.stage, &job.input_state, &output_state, gadget);
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

    pub fn run_stage_sync(
        &mut self,
        stage: usize,
        attempt_budget: usize,
        output_budget: usize,
    ) -> Result<StageRunResult, SynthesisError> {
        self.run_stage_sync_with_inputs(stage, None, attempt_budget, output_budget)
    }

    fn run_stage_sync_with_session(
        &mut self,
        stage: usize,
        attempt_budget: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
    ) -> Result<StageRunResult, SynthesisError> {
        self.run_stage_sync_with_inputs_and_session(
            stage,
            None,
            attempt_budget,
            output_budget,
            session,
        )
    }

    pub fn run_wave_sync(
        &mut self,
        attempt_budget: usize,
        output_budget: usize,
    ) -> Result<WaveRunResult, SynthesisError> {
        let mut session = NullRuntimeSession;
        self.run_wave_sync_with_session(attempt_budget, output_budget, &mut session)
    }

    fn run_wave_sync_with_session(
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
        let scored_paths_before = self.scored_paths.len();
        let last_outputs_before = self.transition_table.unique_output_count(last_stage);
        let target =
            self.run_stage_sync_with_session(target_stage, attempt_budget, output_budget, session)?;
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
            let mut unforwarded = self.transition_table.get_unforwarded_outputs(stage - 1);
            if unforwarded.is_empty() {
                continue;
            }
            unforwarded.sort_by_key(|(state_tuple, _)| state_tuple.clone());

            let forwarded = unforwarded
                .iter()
                .map(|(state_tuple, _)| state_tuple.clone())
                .collect::<Vec<_>>();
            let input_states = unforwarded
                .into_iter()
                .map(|(_, state)| state)
                .collect::<Vec<_>>();

            self.transition_table.mark_forwarded(stage - 1, &forwarded);
            let result = self.run_stage_sync_with_inputs_and_session(
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

        let last_outputs_after = self.transition_table.unique_output_count(last_stage);
        if last_outputs_after > last_outputs_before {
            self.transition_table.reset_unproductive_waves(target_stage);
        } else {
            self.transition_table.record_unproductive_wave(target_stage);
        }

        self.wave_count += 1;
        let scored_paths = self.scored_paths.len() - scored_paths_before;
        Ok(WaveRunResult {
            wave,
            target_stage,
            target,
            propagation,
            discovered_paths: scored_paths,
            scored_paths,
        })
    }

    pub fn run_sync(
        &mut self,
        max_waves: Option<usize>,
        attempt_budget: usize,
        output_budget: usize,
    ) -> Result<WaveSearchResult, SynthesisError> {
        let mut session = NullRuntimeSession;
        self.run_sync_with_session(max_waves, attempt_budget, output_budget, &mut session)
    }

    pub fn run_sync_with_session(
        &mut self,
        max_waves: Option<usize>,
        attempt_budget: usize,
        output_budget: usize,
        session: &mut impl RuntimeSession,
    ) -> Result<WaveSearchResult, SynthesisError> {
        session.on_event(RuntimeEvent::RunStarted {
            stage_count: self.stages.len(),
        });
        for snapshot in self.stage_progress_snapshots() {
            session.on_event(RuntimeEvent::StageUpdated(snapshot));
        }

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
            let wave = self.run_wave_sync_with_session(attempt_budget, output_budget, session)?;
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
            waves.push(wave);
        }

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

        Ok(result)
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

        self.scored_paths.sort_by(|left, right| {
            left.cost()
                .score()
                .total_cmp(&right.cost().score())
                .then_with(|| left.discovery_order().cmp(&right.discovery_order()))
        });
        self.scored_paths.truncate(top_k);
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
        self.run_stage_sync_with_inputs_and_session(
            stage,
            input_states,
            attempt_budget,
            output_budget,
            &mut session,
        )
    }

    fn run_stage_sync_with_inputs_and_session(
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
        self.stage_progress_totals[stage] =
            self.stage_progress_totals[stage].max(attempts_at_start + jobs.len());
        let worker_capacity = usize::from(!jobs.is_empty());
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
        let mut new_transitions = 0;

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
            new_transitions += result.new_transition_count();
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
        })
    }

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
        let (job_sender, job_receiver) = mpsc::channel::<Option<WorkerSynthesisJob>>();
        let job_receiver = Arc::new(Mutex::new(job_receiver));
        let (result_sender, result_receiver) =
            mpsc::channel::<(usize, Result<SynthesisJobOutput, SynthesisError>)>();

        thread::scope(|scope| {
            for _ in 0..worker_count {
                let job_receiver = Arc::clone(&job_receiver);
                let result_sender = result_sender.clone();
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
                        if result_sender
                            .send((index, synthesize_worker_job(config, worker_job)))
                            .is_err()
                        {
                            break;
                        }
                    }
                });
            }
            drop(result_sender);

            let mut next_to_send = 0;
            let mut in_flight = 0;
            let mut stop_sending = false;
            let mut stop_applying = false;
            let mut next_to_apply = 0;
            let mut buffered = BTreeMap::new();
            let mut attempts = 0;
            let mut valid_outputs = 0;
            let mut new_transitions = 0;

            while !session.should_stop() && in_flight < worker_count && next_to_send < jobs.len() {
                self.send_worker_job(&job_sender, next_to_send, &jobs[next_to_send]);
                next_to_send += 1;
                in_flight += 1;
            }
            self.set_stage_worker_state(
                stage,
                in_flight,
                worker_count,
                jobs.len().saturating_sub(next_to_send),
            );
            self.emit_stage_update(stage, session);
            if session.should_stop() {
                stop_sending = true;
                stop_applying = true;
            }

            let mut first_error = None;
            while in_flight > 0 {
                let (index, result) = result_receiver
                    .recv()
                    .expect("worker result channel should stay open while jobs are in flight");
                in_flight -= 1;
                self.set_stage_worker_state(
                    stage,
                    in_flight,
                    worker_count,
                    jobs.len().saturating_sub(next_to_send),
                );

                buffered.insert(index, result);

                while !stop_applying {
                    let Some(result) = buffered.remove(&next_to_apply) else {
                        break;
                    };
                    match result {
                        Ok(output) => {
                            let result = self.apply_synthesis_output(output);
                            attempts += result.attempts();
                            valid_outputs += result.valid_output_count();
                            new_transitions += result.new_transition_count();
                            next_to_apply += 1;
                            self.emit_stage_update(stage, session);
                            if session.should_stop() {
                                stop_sending = true;
                                stop_applying = true;
                                break;
                            }

                            let new_outputs =
                                self.transition_table.unique_output_count(stage) - outputs_at_start;
                            if new_outputs >= output_budget {
                                stop_sending = true;
                                stop_applying = true;
                            }
                        }
                        Err(error) => {
                            first_error = Some(error);
                            stop_sending = true;
                            stop_applying = true;
                        }
                    }
                }

                if session.should_stop() {
                    stop_sending = true;
                    stop_applying = true;
                }

                while !stop_sending && in_flight < worker_count && next_to_send < jobs.len() {
                    if session.should_stop() {
                        stop_sending = true;
                        break;
                    }
                    self.send_worker_job(&job_sender, next_to_send, &jobs[next_to_send]);
                    next_to_send += 1;
                    in_flight += 1;
                }
                self.set_stage_worker_state(
                    stage,
                    in_flight,
                    worker_count,
                    jobs.len().saturating_sub(next_to_send),
                );
                self.emit_stage_update(stage, session);
            }

            for _ in 0..worker_count {
                let _ = job_sender.send(None);
            }

            self.set_stage_worker_state(stage, 0, 0, 0);
            self.emit_stage_update(stage, session);

            if let Some(error) = first_error {
                return Err(error);
            }

            Ok(StageRunResult {
                stage,
                attempts,
                valid_outputs,
                new_outputs: self.transition_table.unique_output_count(stage) - outputs_at_start,
                new_transitions,
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

    fn ordered_outputs_from_stage(&self, stage: usize) -> Vec<VectorState> {
        let mut outputs = self
            .transition_table
            .get_unique_outputs(stage)
            .values()
            .cloned()
            .collect::<Vec<_>>();
        outputs.sort_by_key(VectorState::as_tuple);
        outputs
    }

    fn ordered_inputs_for_stage(
        &self,
        stage: usize,
        input_states: Vec<VectorState>,
    ) -> Vec<VectorState> {
        if stage == 0 {
            return input_states;
        }

        let (mut covered, mut uncovered): (Vec<_>, Vec<_>) =
            input_states.into_iter().partition(|state| {
                self.transition_table
                    .was_input_attempted(stage, &state.as_tuple())
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
            .map(|(left, right)| (*left as u64, *right as u64))
            .collect()
    }
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
            permuted.iter().map(|(top, _)| *top as u64).collect(),
            permuted.iter().map(|(_, bottom)| *bottom as u64).collect(),
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
            .map(|(top, _)| *top as u64)
            .collect(),
        first_stage
            .pairs()
            .iter()
            .map(|(_, bottom)| *bottom as u64)
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
