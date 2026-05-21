use std::collections::{HashMap, HashSet};

use gadget_synth::{PermutationGadget, VectorState};

pub type StateTuple = (Vec<u64>, Vec<u64>);
pub type TransitionKey = (StateTuple, StateTuple);
pub type PathStep = (usize, StateTuple, StateTuple);
pub type CompletePath = Vec<PathStep>;

#[derive(Clone, Debug, PartialEq)]
pub struct StageStats {
    pub attempts: usize,
    pub distinct_outputs: usize,
    pub success_rate: f64,
    pub transition_count: usize,
    pub total_gadgets: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StageData {
    transitions: HashMap<TransitionKey, Vec<PermutationGadget>>,
    unique_outputs: HashMap<StateTuple, VectorState>,
    unique_inputs: HashSet<StateTuple>,
    attempted_pairs: HashSet<(StateTuple, usize)>,
    forwarded_outputs: HashSet<StateTuple>,
    consecutive_zero_budgets: usize,
    unproductive_waves: usize,
    attempts: usize,
    dirty: bool,
}

impl StageData {
    pub fn transitions(&self) -> &HashMap<TransitionKey, Vec<PermutationGadget>> {
        &self.transitions
    }

    pub fn unique_outputs(&self) -> &HashMap<StateTuple, VectorState> {
        &self.unique_outputs
    }

    pub fn unique_inputs(&self) -> &HashSet<StateTuple> {
        &self.unique_inputs
    }

    pub fn attempts(&self) -> usize {
        self.attempts
    }

    pub fn dirty(&self) -> bool {
        self.dirty
    }

    pub fn consecutive_zero_budgets(&self) -> usize {
        self.consecutive_zero_budgets
    }

    pub fn unproductive_waves(&self) -> usize {
        self.unproductive_waves
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionTable {
    stages: Vec<StageData>,
}

impl TransitionTable {
    pub fn new(num_stages: usize) -> Self {
        Self {
            stages: vec![StageData::default(); num_stages],
        }
    }

    pub fn stage(&self, stage: usize) -> &StageData {
        &self.stages[stage]
    }

    pub fn stage_mut(&mut self, stage: usize) -> &mut StageData {
        &mut self.stages[stage]
    }

    pub fn stages(&self) -> &[StageData] {
        &self.stages
    }

    pub fn add_transition(
        &mut self,
        stage: usize,
        input_state: &VectorState,
        output_state: &VectorState,
        gadget: PermutationGadget,
    ) -> bool {
        let stage_data = &mut self.stages[stage];
        let input_key = input_state.as_tuple();
        let output_key = output_state.as_tuple();
        let transition_key = (input_key.clone(), output_key.clone());

        let gadgets = stage_data.transitions.entry(transition_key).or_default();
        if gadgets.contains(&gadget) {
            return false;
        }

        gadgets.push(gadget);
        stage_data.unique_inputs.insert(input_key);
        stage_data
            .unique_outputs
            .entry(output_key)
            .or_insert_with(|| output_state.clone());
        stage_data.dirty = true;
        true
    }

    pub fn record_attempt(&mut self, stage: usize, count: usize) {
        let stage_data = &mut self.stages[stage];
        stage_data.attempts += count;
        stage_data.dirty = true;
    }

    pub fn record_attempted_pair(
        &mut self,
        stage: usize,
        input_tuple: &StateTuple,
        candidate_index: usize,
    ) {
        let stage_data = &mut self.stages[stage];
        stage_data
            .attempted_pairs
            .insert((input_tuple.clone(), candidate_index));
        stage_data.dirty = true;
    }

    pub fn was_attempted(
        &self,
        stage: usize,
        input_tuple: &StateTuple,
        candidate_index: usize,
    ) -> bool {
        self.stages[stage]
            .attempted_pairs
            .contains(&(input_tuple.clone(), candidate_index))
    }

    pub fn was_input_attempted(&self, stage: usize, input_tuple: &StateTuple) -> bool {
        self.stages[stage]
            .attempted_pairs
            .iter()
            .any(|(attempted_input, _)| attempted_input == input_tuple)
    }

    pub fn get_transitions(
        &self,
        stage: usize,
        input_tuple: &StateTuple,
    ) -> HashMap<StateTuple, Vec<PermutationGadget>> {
        self.stages[stage]
            .transitions
            .iter()
            .filter_map(|((input, output), gadgets)| {
                (input == input_tuple).then(|| (output.clone(), gadgets.clone()))
            })
            .collect()
    }

    pub fn get_all_transitions(
        &self,
        stage: usize,
    ) -> &HashMap<TransitionKey, Vec<PermutationGadget>> {
        &self.stages[stage].transitions
    }

    pub fn get_unique_outputs(&self, stage: usize) -> &HashMap<StateTuple, VectorState> {
        &self.stages[stage].unique_outputs
    }

    pub fn unique_output_count(&self, stage: usize) -> usize {
        self.stages[stage].unique_outputs.len()
    }

    pub fn get_unforwarded_outputs(&self, stage: usize) -> Vec<(StateTuple, VectorState)> {
        let stage_data = &self.stages[stage];
        stage_data
            .unique_outputs
            .iter()
            .filter_map(|(state_tuple, state)| {
                (!stage_data.forwarded_outputs.contains(state_tuple))
                    .then(|| (state_tuple.clone(), state.clone()))
            })
            .collect()
    }

    pub fn mark_forwarded(&mut self, stage: usize, output_tuples: &[StateTuple]) {
        self.stages[stage]
            .forwarded_outputs
            .extend(output_tuples.iter().cloned());
    }

    pub fn record_zero_output_budget(&mut self, stage: usize) {
        let stage_data = &mut self.stages[stage];
        stage_data.consecutive_zero_budgets += 1;
        stage_data.dirty = true;
    }

    pub fn reset_zero_output_budgets(&mut self, stage: usize) {
        let stage_data = &mut self.stages[stage];
        stage_data.consecutive_zero_budgets = 0;
        stage_data.dirty = true;
    }

    pub fn record_unproductive_wave(&mut self, stage: usize) {
        let stage_data = &mut self.stages[stage];
        stage_data.unproductive_waves += 1;
        stage_data.dirty = true;
    }

    pub fn reset_unproductive_waves(&mut self, stage: usize) {
        let stage_data = &mut self.stages[stage];
        stage_data.unproductive_waves = 0;
        stage_data.dirty = true;
    }

    pub fn stage_stats(&self, stage: usize) -> StageStats {
        let stage_data = &self.stages[stage];
        let transition_count = stage_data.transitions.len();
        let total_gadgets = stage_data.transitions.values().map(Vec::len).sum::<usize>();
        let distinct_outputs = stage_data.unique_outputs.len();
        let success_rate = if stage_data.attempts == 0 {
            0.0
        } else {
            distinct_outputs as f64 / stage_data.attempts as f64
        };

        StageStats {
            attempts: stage_data.attempts,
            distinct_outputs,
            success_rate,
            transition_count,
            total_gadgets,
        }
    }

    pub fn enumerate_complete_paths(
        &self,
        start: Option<&StateTuple>,
        max_paths: Option<usize>,
        exclude_paths: Option<&HashSet<CompletePath>>,
    ) -> Vec<CompletePath> {
        if self.stages.is_empty() {
            return Vec::new();
        }

        let exclude = exclude_paths;
        let mut results = Vec::new();
        let mut current_path = Vec::new();
        self.enumerate_complete_paths_from(
            0,
            start.cloned(),
            &mut current_path,
            &mut results,
            max_paths,
            exclude,
        );
        results
    }

    fn enumerate_complete_paths_from(
        &self,
        stage: usize,
        required_input: Option<StateTuple>,
        current_path: &mut CompletePath,
        results: &mut Vec<CompletePath>,
        max_paths: Option<usize>,
        exclude_paths: Option<&HashSet<CompletePath>>,
    ) {
        if max_paths.is_some_and(|limit| results.len() >= limit) {
            return;
        }

        if stage >= self.stages.len() {
            if !exclude_paths.is_some_and(|exclude| exclude.contains(current_path)) {
                results.push(current_path.clone());
            }
            return;
        }

        let transitions = self.stages[stage]
            .transitions
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let mut transitions = transitions;
        transitions.sort();
        for (input, output) in transitions {
            if required_input
                .as_ref()
                .is_some_and(|required| required != &input)
            {
                continue;
            }

            current_path.push((stage, input.clone(), output.clone()));
            self.enumerate_complete_paths_from(
                stage + 1,
                Some(output),
                current_path,
                results,
                max_paths,
                exclude_paths,
            );
            current_path.pop();

            if max_paths.is_some_and(|limit| results.len() >= limit) {
                return;
            }
        }
    }

    pub fn trace_paths_ending_with(
        &self,
        stage: usize,
        input_tuple: &StateTuple,
        output_tuple: &StateTuple,
        max_paths: Option<usize>,
        exclude_paths: Option<&HashSet<CompletePath>>,
    ) -> Vec<CompletePath> {
        let anchor_key = (input_tuple.clone(), output_tuple.clone());
        if !self.stages[stage].transitions.contains_key(&anchor_key) {
            return Vec::new();
        }

        let anchor_step = (stage, input_tuple.clone(), output_tuple.clone());
        if stage == 0 {
            let single = vec![anchor_step];
            return if exclude_paths.is_some_and(|exclude| exclude.contains(&single)) {
                Vec::new()
            } else {
                vec![single]
            };
        }

        let mut results = Vec::new();
        let mut suffix = vec![anchor_step];
        self.trace_paths_back(
            stage - 1,
            input_tuple.clone(),
            &mut suffix,
            &mut results,
            max_paths,
            exclude_paths,
        );
        results
    }

    fn trace_paths_back(
        &self,
        current_stage: usize,
        required_output: StateTuple,
        suffix: &mut CompletePath,
        results: &mut Vec<CompletePath>,
        max_paths: Option<usize>,
        exclude_paths: Option<&HashSet<CompletePath>>,
    ) {
        if max_paths.is_some_and(|limit| results.len() >= limit) {
            return;
        }

        let transitions = self.stages[current_stage]
            .transitions
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let mut transitions = transitions;
        transitions.sort();
        for (input, output) in transitions {
            if output != required_output {
                continue;
            }

            suffix.push((current_stage, input.clone(), output));
            if current_stage == 0 {
                let mut candidate = suffix.clone();
                candidate.reverse();
                if !exclude_paths.is_some_and(|exclude| exclude.contains(&candidate)) {
                    results.push(candidate);
                }
            } else {
                self.trace_paths_back(
                    current_stage - 1,
                    input,
                    suffix,
                    results,
                    max_paths,
                    exclude_paths,
                );
            }
            suffix.pop();

            if max_paths.is_some_and(|limit| results.len() >= limit) {
                return;
            }
        }
    }
}
