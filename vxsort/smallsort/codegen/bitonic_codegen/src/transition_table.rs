use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

pub use gadget_synth::LaneLabel;
use gadget_synth::{PermutationGadget, VectorState};
use serde::{Deserialize, Serialize};

use crate::stable_vec::{StableVec, StableVecSnapshot};

pub type StateTuple = (Vec<LaneLabel>, Vec<LaneLabel>);
pub type TransitionKey = (StateTuple, StateTuple);
pub type GadgetList = StableVec<PermutationGadget, 1, 16>;
pub type GadgetListSnapshot = StableVecSnapshot<PermutationGadget, 1, 16>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct StateId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TransitionIndex(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PathId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct GadgetIndex(pub u16);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct TransitionRef {
    pub stage: u16,
    pub transition: TransitionIndex,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateInterner {
    lanes_per_side: Option<usize>,
    states: Vec<VectorState>,
    by_state: HashMap<VectorState, StateId>,
}

impl StateInterner {
    pub fn new(lanes_per_side: usize) -> Self {
        Self {
            lanes_per_side: Some(lanes_per_side),
            states: Vec::new(),
            by_state: HashMap::new(),
        }
    }

    pub fn inferred() -> Self {
        Self {
            lanes_per_side: None,
            states: Vec::new(),
            by_state: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn lanes_per_side(&self) -> Option<usize> {
        self.lanes_per_side
    }

    pub fn intern(&mut self, state: VectorState) -> StateId {
        self.ensure_shape(&state);
        if let Some(id) = self.by_state.get(&state) {
            return *id;
        }

        let id = StateId(
            self.states
                .len()
                .try_into()
                .expect("state table should fit in u32"),
        );
        self.states.push(state.clone());
        self.by_state.insert(state, id);
        id
    }

    pub fn lookup(&self, state: &VectorState) -> Option<StateId> {
        self.by_state.get(state).copied()
    }

    pub fn get(&self, id: StateId) -> &VectorState {
        &self.states[id.0 as usize]
    }

    fn ensure_shape(&mut self, state: &VectorState) {
        assert_eq!(
            state.top().len(),
            state.bottom().len(),
            "state must contain equal top/bottom lane counts"
        );
        let lanes_per_side = state.lanes_per_side();
        match self.lanes_per_side {
            Some(expected) => assert_eq!(
                expected, lanes_per_side,
                "state lane count changed within transition table"
            ),
            None => self.lanes_per_side = Some(lanes_per_side),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct CompletePath {
    transitions: Box<[TransitionIndex]>,
}

impl CompletePath {
    pub fn new(transitions: Vec<TransitionIndex>) -> Self {
        Self {
            transitions: transitions.into_boxed_slice(),
        }
    }

    pub fn transitions(&self) -> &[TransitionIndex] {
        &self.transitions
    }

    pub fn transition_at_stage(&self, stage: usize) -> TransitionIndex {
        self.transitions[stage]
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, TransitionIndex)> + '_ {
        self.transitions.iter().copied().enumerate()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PathRegistration {
    pub id: PathId,
    pub was_new: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PathRegistry {
    paths: Vec<CompletePath>,
    path_by_transitions: HashMap<CompletePath, PathId>,
    paths_by_transition: HashMap<TransitionRef, Vec<PathId>>,
    empty_paths: Vec<PathId>,
}

impl PathRegistry {
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    pub fn register(&mut self, path: CompletePath) -> PathRegistration {
        if let Some(id) = self.path_by_transitions.get(&path).copied() {
            return PathRegistration { id, was_new: false };
        }

        let id = PathId(
            self.paths
                .len()
                .try_into()
                .expect("path registry should fit in u32"),
        );
        for (stage, transition) in path.iter() {
            let transition_ref = TransitionRef {
                stage: stage
                    .try_into()
                    .expect("stage index should fit in transition ref"),
                transition,
            };
            self.paths_by_transition
                .entry(transition_ref)
                .or_default()
                .push(id);
        }
        self.path_by_transitions.insert(path.clone(), id);
        self.paths.push(path);

        PathRegistration { id, was_new: true }
    }

    pub fn path(&self, id: PathId) -> &CompletePath {
        &self.paths[id.0 as usize]
    }

    pub fn paths_for_transition(&self, transition: TransitionRef) -> &[PathId] {
        self.paths_by_transition
            .get(&transition)
            .unwrap_or(&self.empty_paths)
    }

    pub fn indexed_transition_count(&self) -> usize {
        self.paths_by_transition.len()
    }

    pub fn indexed_path_ref_count(&self) -> usize {
        self.paths_by_transition
            .values()
            .map(|path_ids| path_ids.len())
            .sum()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct TransitionRecord {
    input: StateId,
    output: StateId,
    gadgets: Arc<GadgetList>,
}

impl Clone for TransitionRecord {
    fn clone(&self) -> Self {
        let gadgets = Arc::new(GadgetList::new());
        for gadget in self.gadgets().iter().cloned() {
            gadgets.push(gadget);
        }
        Self {
            input: self.input,
            output: self.output,
            gadgets,
        }
    }
}

impl TransitionRecord {
    pub fn input(&self) -> StateId {
        self.input
    }

    pub fn output(&self) -> StateId {
        self.output
    }

    pub fn gadgets(&self) -> GadgetListSnapshot {
        self.gadgets.snapshot()
    }

    pub fn gadget_count(&self) -> usize {
        self.gadgets.len()
    }

    pub fn gadget(&self, index: GadgetIndex) -> &PermutationGadget {
        self.gadgets.get(index.0 as usize)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionInsertResult {
    pub transition: TransitionRef,
    pub transition_was_new: bool,
    pub gadget_was_new: bool,
    pub gadget_index: Option<GadgetIndex>,
}

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
    transitions: Vec<TransitionRecord>,
    transition_by_pair: HashMap<(StateId, StateId), TransitionIndex>,
    transitions_by_input: HashMap<StateId, Vec<TransitionIndex>>,
    transitions_by_output: HashMap<StateId, Vec<TransitionIndex>>,
    unique_outputs: HashSet<StateId>,
    unique_inputs: HashSet<StateId>,
    attempted_pairs: HashSet<(StateId, usize)>,
    forwarded_outputs: HashSet<StateId>,
    consecutive_zero_budgets: usize,
    unproductive_waves: usize,
    attempts: usize,
    dirty: bool,
}

impl StageData {
    pub fn transition_records(&self) -> &[TransitionRecord] {
        &self.transitions
    }

    pub fn unique_output_ids(&self) -> &HashSet<StateId> {
        &self.unique_outputs
    }

    pub fn unique_input_ids(&self) -> &HashSet<StateId> {
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

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    pub fn total_gadget_count(&self) -> usize {
        self.transitions
            .iter()
            .map(TransitionRecord::gadget_count)
            .sum()
    }

    pub fn attempted_pair_count(&self) -> usize {
        self.attempted_pairs.len()
    }

    pub fn forwarded_output_count(&self) -> usize {
        self.forwarded_outputs.len()
    }

    pub fn transitions_by_input_entry_count(&self) -> usize {
        self.transitions_by_input.len()
    }

    pub fn transitions_by_input_ref_count(&self) -> usize {
        self.transitions_by_input
            .values()
            .map(|transitions| transitions.len())
            .sum()
    }

    pub fn transitions_by_output_entry_count(&self) -> usize {
        self.transitions_by_output.len()
    }

    pub fn transitions_by_output_ref_count(&self) -> usize {
        self.transitions_by_output
            .values()
            .map(|transitions| transitions.len())
            .sum()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionTable {
    states: StateInterner,
    stages: Box<[StageData]>,
}

#[derive(Debug)]
pub struct ResolvedPathStep {
    pub stage: usize,
    pub transition: TransitionRef,
    pub input: StateId,
    pub output: StateId,
    pub input_state: StateTuple,
    pub output_state: StateTuple,
    pub gadgets: GadgetListSnapshot,
}

impl TransitionTable {
    pub fn new(num_stages: usize) -> Self {
        Self {
            states: StateInterner::inferred(),
            stages: vec![StageData::default(); num_stages].into_boxed_slice(),
        }
    }

    pub fn new_with_lanes(num_stages: usize, lanes_per_side: usize) -> Self {
        Self {
            states: StateInterner::new(lanes_per_side),
            stages: vec![StageData::default(); num_stages].into_boxed_slice(),
        }
    }

    pub fn states(&self) -> &StateInterner {
        &self.states
    }

    pub fn state(&self, id: StateId) -> &VectorState {
        self.states.get(id)
    }

    pub fn state_as_zero_based_tuple(&self, id: StateId) -> StateTuple {
        self.state(id).to_zero_based_tuple()
    }

    pub fn state_as_one_based_tuple(&self, id: StateId) -> StateTuple {
        self.state(id).to_one_based_tuple()
    }

    pub fn state_as_vector_state(&self, id: StateId) -> VectorState {
        self.state(id).clone()
    }

    pub fn intern_zero_based_state(&mut self, state: &VectorState) -> StateId {
        self.states.intern(state.clone())
    }

    pub fn intern_zero_based_tuple(&mut self, state: &StateTuple) -> StateId {
        let state = VectorState::try_from_zero_based_labels(&state.0, &state.1)
            .expect("state tuple labels should fit in compact state storage");
        self.states.intern(state)
    }

    pub fn intern_one_based_tuple(&mut self, state: &StateTuple) -> StateId {
        let state = VectorState::try_from_one_based_labels(&state.0, &state.1)
            .expect("one-based state tuple labels should fit in compact state storage");
        self.states.intern(state)
    }

    pub fn lookup_zero_based_tuple(&self, state: &StateTuple) -> Option<StateId> {
        let state = VectorState::try_from_zero_based_labels(&state.0, &state.1).ok()?;
        self.states.lookup(&state)
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

    pub fn transition(&self, reference: TransitionRef) -> &TransitionRecord {
        &self.stages[usize::from(reference.stage)].transitions[reference.transition.0 as usize]
    }

    pub fn transition_gadgets(&self, reference: TransitionRef) -> GadgetListSnapshot {
        self.transition(reference).gadgets()
    }

    pub fn transition_ref_for_ids(
        &self,
        stage: usize,
        input: StateId,
        output: StateId,
    ) -> Option<TransitionRef> {
        self.stages[stage]
            .transition_by_pair
            .get(&(input, output))
            .copied()
            .map(|transition| TransitionRef {
                stage: stage
                    .try_into()
                    .expect("stage index should fit in u16 transition reference"),
                transition,
            })
    }

    pub fn transition_ref_for_zero_based_tuples(
        &self,
        stage: usize,
        input_tuple: &StateTuple,
        output_tuple: &StateTuple,
    ) -> Option<TransitionRef> {
        let input = self.lookup_zero_based_tuple(input_tuple)?;
        let output = self.lookup_zero_based_tuple(output_tuple)?;
        self.transition_ref_for_ids(stage, input, output)
    }

    pub fn transition_ref_for_one_based_tuples(
        &self,
        stage: usize,
        input_tuple: &StateTuple,
        output_tuple: &StateTuple,
    ) -> Option<TransitionRef> {
        let input = VectorState::try_from_one_based_labels(&input_tuple.0, &input_tuple.1).ok()?;
        let output =
            VectorState::try_from_one_based_labels(&output_tuple.0, &output_tuple.1).ok()?;
        let input = self.states.lookup(&input)?;
        let output = self.states.lookup(&output)?;
        self.transition_ref_for_ids(stage, input, output)
    }

    pub fn add_transition_by_id(
        &mut self,
        stage: usize,
        input: StateId,
        output: VectorState,
        gadget: PermutationGadget,
    ) -> TransitionInsertResult {
        let output = self.states.intern(output);
        self.add_transition_ids(stage, input, output, gadget)
    }

    pub fn add_transition_ids(
        &mut self,
        stage: usize,
        input: StateId,
        output: StateId,
        gadget: PermutationGadget,
    ) -> TransitionInsertResult {
        let stage_data = &mut self.stages[stage];
        let stage_ref = stage
            .try_into()
            .expect("stage index should fit in u16 transition reference");

        if let Some(transition) = stage_data.transition_by_pair.get(&(input, output)).copied() {
            let record = &mut stage_data.transitions[transition.0 as usize];
            if let Some(existing_index) = record
                .gadgets
                .snapshot()
                .iter()
                .position(|existing| existing == &gadget)
            {
                return TransitionInsertResult {
                    transition: TransitionRef {
                        stage: stage_ref,
                        transition,
                    },
                    transition_was_new: false,
                    gadget_was_new: false,
                    gadget_index: Some(GadgetIndex(
                        existing_index
                            .try_into()
                            .expect("gadget index should fit in u16"),
                    )),
                };
            }

            let gadget_index = GadgetIndex(
                record
                    .gadgets
                    .len()
                    .try_into()
                    .expect("gadget index should fit in u16"),
            );
            record.gadgets.push(gadget);
            stage_data.dirty = true;
            return TransitionInsertResult {
                transition: TransitionRef {
                    stage: stage_ref,
                    transition,
                },
                transition_was_new: false,
                gadget_was_new: true,
                gadget_index: Some(gadget_index),
            };
        }

        let transition = TransitionIndex(
            stage_data
                .transitions
                .len()
                .try_into()
                .expect("transition index should fit in u32"),
        );
        stage_data.transitions.push(TransitionRecord {
            input,
            output,
            gadgets: {
                let gadgets = Arc::new(GadgetList::new());
                gadgets.push(gadget);
                gadgets
            },
        });
        stage_data
            .transition_by_pair
            .insert((input, output), transition);
        stage_data
            .transitions_by_input
            .entry(input)
            .or_default()
            .push(transition);
        stage_data
            .transitions_by_output
            .entry(output)
            .or_default()
            .push(transition);
        stage_data.unique_inputs.insert(input);
        stage_data.unique_outputs.insert(output);
        stage_data.dirty = true;

        TransitionInsertResult {
            transition: TransitionRef {
                stage: stage_ref,
                transition,
            },
            transition_was_new: true,
            gadget_was_new: true,
            gadget_index: Some(GadgetIndex(0)),
        }
    }

    pub fn add_transition(
        &mut self,
        stage: usize,
        input_state: &VectorState,
        output_state: &VectorState,
        gadget: PermutationGadget,
    ) -> bool {
        let input = self.intern_zero_based_state(input_state);
        self.add_transition_by_id(stage, input, output_state.clone(), gadget)
            .gadget_was_new
    }

    pub fn add_transition_one_based_tuples(
        &mut self,
        stage: usize,
        input_tuple: &StateTuple,
        output_tuple: &StateTuple,
        gadget: PermutationGadget,
    ) -> TransitionInsertResult {
        let input = VectorState::try_from_one_based_labels(&input_tuple.0, &input_tuple.1)
            .expect("one-based input state labels should fit in compact state storage");
        let output = VectorState::try_from_one_based_labels(&output_tuple.0, &output_tuple.1)
            .expect("one-based output state labels should fit in compact state storage");
        let input = self.states.intern(input);
        self.add_transition_by_id(stage, input, output, gadget)
    }

    pub fn record_attempt(&mut self, stage: usize, count: usize) {
        let stage_data = &mut self.stages[stage];
        stage_data.attempts += count;
        stage_data.dirty = true;
    }

    pub fn record_attempted_pair_by_id(
        &mut self,
        stage: usize,
        input: StateId,
        candidate_index: usize,
    ) {
        let stage_data = &mut self.stages[stage];
        stage_data.attempted_pairs.insert((input, candidate_index));
        stage_data.dirty = true;
    }

    pub fn record_attempted_pair(
        &mut self,
        stage: usize,
        input_tuple: &StateTuple,
        candidate_index: usize,
    ) {
        let input = self.intern_zero_based_tuple(input_tuple);
        self.record_attempted_pair_by_id(stage, input, candidate_index);
    }

    pub fn was_attempted_by_id(
        &self,
        stage: usize,
        input: StateId,
        candidate_index: usize,
    ) -> bool {
        self.stages[stage]
            .attempted_pairs
            .contains(&(input, candidate_index))
    }

    pub fn was_attempted(
        &self,
        stage: usize,
        input_tuple: &StateTuple,
        candidate_index: usize,
    ) -> bool {
        let Some(input) = self.lookup_zero_based_tuple(input_tuple) else {
            return false;
        };
        self.was_attempted_by_id(stage, input, candidate_index)
    }

    pub fn was_input_attempted_by_id(&self, stage: usize, input: StateId) -> bool {
        self.stages[stage]
            .attempted_pairs
            .iter()
            .any(|(attempted_input, _)| *attempted_input == input)
    }

    pub fn was_input_attempted(&self, stage: usize, input_tuple: &StateTuple) -> bool {
        let Some(input) = self.lookup_zero_based_tuple(input_tuple) else {
            return false;
        };
        self.was_input_attempted_by_id(stage, input)
    }

    pub fn get_transitions(
        &self,
        stage: usize,
        input_tuple: &StateTuple,
    ) -> HashMap<StateTuple, Vec<PermutationGadget>> {
        let Some(input) = self.lookup_zero_based_tuple(input_tuple) else {
            return HashMap::new();
        };
        let mut transitions = HashMap::new();
        for transition in self.stages[stage]
            .transitions_by_input
            .get(&input)
            .into_iter()
            .flatten()
        {
            let record = &self.stages[stage].transitions[transition.0 as usize];
            transitions.insert(
                self.state_as_zero_based_tuple(record.output),
                record.gadgets().to_vec(),
            );
        }
        transitions
    }

    pub fn get_all_transitions(
        &self,
        stage: usize,
    ) -> HashMap<TransitionKey, Vec<PermutationGadget>> {
        self.stages[stage]
            .transitions
            .iter()
            .map(|record| {
                (
                    (
                        self.state_as_zero_based_tuple(record.input),
                        self.state_as_zero_based_tuple(record.output),
                    ),
                    record.gadgets().to_vec(),
                )
            })
            .collect()
    }

    pub fn get_unique_outputs(&self, stage: usize) -> HashMap<StateTuple, VectorState> {
        self.stages[stage]
            .unique_outputs
            .iter()
            .map(|state_id| {
                let state = self.state_as_zero_based_tuple(*state_id);
                (state, self.state_as_vector_state(*state_id))
            })
            .collect()
    }

    pub fn get_unique_output_ids(&self, stage: usize) -> Vec<(StateId, VectorState)> {
        self.stages[stage]
            .unique_outputs
            .iter()
            .map(|state_id| (*state_id, self.state_as_vector_state(*state_id)))
            .collect()
    }

    pub fn unique_output_count(&self, stage: usize) -> usize {
        self.stages[stage].unique_outputs.len()
    }

    pub fn get_unforwarded_output_ids(&self, stage: usize) -> Vec<(StateId, VectorState)> {
        let stage_data = &self.stages[stage];
        stage_data
            .unique_outputs
            .iter()
            .filter(|state_id| !stage_data.forwarded_outputs.contains(state_id))
            .map(|state_id| (*state_id, self.state_as_vector_state(*state_id)))
            .collect()
    }

    pub fn get_unforwarded_outputs(&self, stage: usize) -> Vec<(StateTuple, VectorState)> {
        self.get_unforwarded_output_ids(stage)
            .into_iter()
            .map(|(state_id, state)| (self.state_as_zero_based_tuple(state_id), state))
            .collect()
    }

    pub fn mark_forwarded_ids(&mut self, stage: usize, output_ids: &[StateId]) {
        self.stages[stage]
            .forwarded_outputs
            .extend(output_ids.iter().copied());
    }

    pub fn mark_forwarded(&mut self, stage: usize, output_tuples: &[StateTuple]) {
        let output_ids = output_tuples
            .iter()
            .filter_map(|state| self.lookup_zero_based_tuple(state))
            .collect::<Vec<_>>();
        self.mark_forwarded_ids(stage, &output_ids);
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
        let total_gadgets = stage_data
            .transitions
            .iter()
            .map(|transition| transition.gadgets.len())
            .sum::<usize>();
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
        let start = start.and_then(|state| self.lookup_zero_based_tuple(state));
        let mut results = Vec::new();
        let mut current_path = Vec::new();
        self.enumerate_complete_paths_from(
            0,
            start,
            &mut current_path,
            &mut results,
            max_paths,
            exclude_paths,
        );
        results
    }

    fn enumerate_complete_paths_from(
        &self,
        stage: usize,
        required_input: Option<StateId>,
        current_path: &mut Vec<TransitionIndex>,
        results: &mut Vec<CompletePath>,
        max_paths: Option<usize>,
        exclude_paths: Option<&HashSet<CompletePath>>,
    ) {
        if max_paths.is_some_and(|limit| results.len() >= limit) {
            return;
        }

        if stage >= self.stages.len() {
            let candidate = CompletePath::new(current_path.clone());
            if !exclude_paths.is_some_and(|exclude| exclude.contains(&candidate)) {
                results.push(candidate);
            }
            return;
        }

        let mut transitions = match required_input {
            Some(input) => self.stages[stage]
                .transitions_by_input
                .get(&input)
                .cloned()
                .unwrap_or_default(),
            None => (0..self.stages[stage].transitions.len())
                .map(|idx| TransitionIndex(idx.try_into().expect("transition index fits u32")))
                .collect(),
        };
        self.sort_transition_indexes(stage, &mut transitions);

        for transition in transitions {
            let output = self.stages[stage].transitions[transition.0 as usize].output;
            current_path.push(transition);
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

    pub fn trace_paths_ending_at(
        &self,
        anchor: TransitionRef,
        max_paths: Option<usize>,
        exclude_paths: Option<&HashSet<CompletePath>>,
    ) -> Vec<CompletePath> {
        let anchor_record = self.transition(anchor);
        if anchor.stage == 0 {
            let single = CompletePath::new(vec![anchor.transition]);
            return if exclude_paths.is_some_and(|exclude| exclude.contains(&single)) {
                Vec::new()
            } else {
                vec![single]
            };
        }

        let mut results = Vec::new();
        let mut suffix = vec![anchor.transition];
        self.trace_paths_back(
            usize::from(anchor.stage) - 1,
            anchor_record.input,
            &mut suffix,
            &mut results,
            max_paths,
            exclude_paths,
        );
        results
    }

    pub fn trace_paths_ending_with(
        &self,
        stage: usize,
        input_tuple: &StateTuple,
        output_tuple: &StateTuple,
        max_paths: Option<usize>,
        exclude_paths: Option<&HashSet<CompletePath>>,
    ) -> Vec<CompletePath> {
        let Some(anchor) =
            self.transition_ref_for_zero_based_tuples(stage, input_tuple, output_tuple)
        else {
            return Vec::new();
        };
        self.trace_paths_ending_at(anchor, max_paths, exclude_paths)
    }

    pub fn trace_paths_through(
        &self,
        anchor: TransitionRef,
        max_paths: Option<usize>,
        exclude_paths: Option<&HashSet<CompletePath>>,
    ) -> Vec<CompletePath> {
        if self.stages.is_empty() {
            return Vec::new();
        }

        let anchor_stage = usize::from(anchor.stage);
        let anchor_record = self.transition(anchor);
        let prefixes = if anchor_stage == 0 {
            vec![Vec::new()]
        } else {
            let mut prefixes = Vec::new();
            let mut current = Vec::new();
            self.collect_prefix_paths(
                anchor_stage - 1,
                anchor_record.input,
                &mut current,
                &mut prefixes,
                max_paths,
            );
            prefixes
        };
        if prefixes.is_empty() {
            return Vec::new();
        }

        let suffixes = if anchor_stage + 1 >= self.stages.len() {
            vec![Vec::new()]
        } else {
            let mut suffixes = Vec::new();
            let mut current = Vec::new();
            self.collect_suffix_paths(
                anchor_stage + 1,
                anchor_record.output,
                &mut current,
                &mut suffixes,
                max_paths,
            );
            suffixes
        };
        if suffixes.is_empty() {
            return Vec::new();
        }

        let mut results = Vec::new();
        for prefix in &prefixes {
            for suffix in &suffixes {
                if max_paths.is_some_and(|limit| results.len() >= limit) {
                    return results;
                }
                let mut transitions = Vec::with_capacity(prefix.len() + 1 + suffix.len());
                transitions.extend(prefix.iter().copied());
                transitions.push(anchor.transition);
                transitions.extend(suffix.iter().copied());
                let candidate = CompletePath::new(transitions);
                if !exclude_paths.is_some_and(|exclude| exclude.contains(&candidate)) {
                    results.push(candidate);
                }
            }
        }
        results
    }

    fn trace_paths_back(
        &self,
        current_stage: usize,
        required_output: StateId,
        suffix: &mut Vec<TransitionIndex>,
        results: &mut Vec<CompletePath>,
        max_paths: Option<usize>,
        exclude_paths: Option<&HashSet<CompletePath>>,
    ) {
        if max_paths.is_some_and(|limit| results.len() >= limit) {
            return;
        }

        let mut transitions = self.stages[current_stage]
            .transitions_by_output
            .get(&required_output)
            .cloned()
            .unwrap_or_default();
        self.sort_transition_indexes(current_stage, &mut transitions);

        for transition in transitions {
            let input = self.stages[current_stage].transitions[transition.0 as usize].input;
            suffix.push(transition);
            if current_stage == 0 {
                let mut candidate = suffix.clone();
                candidate.reverse();
                let candidate = CompletePath::new(candidate);
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

    fn collect_prefix_paths(
        &self,
        current_stage: usize,
        required_output: StateId,
        current: &mut Vec<TransitionIndex>,
        results: &mut Vec<Vec<TransitionIndex>>,
        max_paths: Option<usize>,
    ) {
        if max_paths.is_some_and(|limit| results.len() >= limit) {
            return;
        }

        let mut transitions = self.stages[current_stage]
            .transitions_by_output
            .get(&required_output)
            .cloned()
            .unwrap_or_default();
        self.sort_transition_indexes(current_stage, &mut transitions);

        for transition in transitions {
            let input = self.stages[current_stage].transitions[transition.0 as usize].input;
            current.push(transition);
            if current_stage == 0 {
                let mut path = current.clone();
                path.reverse();
                results.push(path);
            } else {
                self.collect_prefix_paths(current_stage - 1, input, current, results, max_paths);
            }
            current.pop();

            if max_paths.is_some_and(|limit| results.len() >= limit) {
                return;
            }
        }
    }

    fn collect_suffix_paths(
        &self,
        current_stage: usize,
        required_input: StateId,
        current: &mut Vec<TransitionIndex>,
        results: &mut Vec<Vec<TransitionIndex>>,
        max_paths: Option<usize>,
    ) {
        if max_paths.is_some_and(|limit| results.len() >= limit) {
            return;
        }

        if current_stage >= self.stages.len() {
            results.push(current.clone());
            return;
        }

        let mut transitions = self.stages[current_stage]
            .transitions_by_input
            .get(&required_input)
            .cloned()
            .unwrap_or_default();
        self.sort_transition_indexes(current_stage, &mut transitions);

        for transition in transitions {
            let output = self.stages[current_stage].transitions[transition.0 as usize].output;
            current.push(transition);
            self.collect_suffix_paths(current_stage + 1, output, current, results, max_paths);
            current.pop();

            if max_paths.is_some_and(|limit| results.len() >= limit) {
                return;
            }
        }
    }

    pub fn resolve_complete_path(&self, path: &CompletePath) -> Vec<ResolvedPathStep> {
        path.iter()
            .map(|(stage, transition)| {
                let reference = TransitionRef {
                    stage: stage
                        .try_into()
                        .expect("stage index should fit in u16 transition reference"),
                    transition,
                };
                let record = self.transition(reference);
                ResolvedPathStep {
                    stage,
                    transition: reference,
                    input: record.input,
                    output: record.output,
                    input_state: self.state_as_zero_based_tuple(record.input),
                    output_state: self.state_as_zero_based_tuple(record.output),
                    gadgets: record.gadgets(),
                }
            })
            .collect()
    }

    pub fn complete_path_from_zero_based_steps(
        &self,
        steps: &[(usize, StateTuple, StateTuple)],
    ) -> Option<CompletePath> {
        steps
            .iter()
            .map(|(stage, input, output)| {
                self.transition_ref_for_zero_based_tuples(*stage, input, output)
                    .map(|reference| reference.transition)
            })
            .collect::<Option<Vec<_>>>()
            .map(CompletePath::new)
    }

    pub fn complete_path_from_one_based_steps(
        &self,
        steps: &[(usize, StateTuple, StateTuple)],
    ) -> Option<CompletePath> {
        steps
            .iter()
            .map(|(stage, input, output)| {
                self.transition_ref_for_one_based_tuples(*stage, input, output)
                    .map(|reference| reference.transition)
            })
            .collect::<Option<Vec<_>>>()
            .map(CompletePath::new)
    }

    fn sort_transition_indexes(&self, stage: usize, transitions: &mut [TransitionIndex]) {
        transitions.sort_by(|left, right| {
            let left_record = &self.stages[stage].transitions[left.0 as usize];
            let right_record = &self.stages[stage].transitions[right.0 as usize];
            self.state(left_record.input)
                .cmp(self.state(right_record.input))
                .then_with(|| {
                    self.state(left_record.output)
                        .cmp(self.state(right_record.output))
                })
                .then_with(|| left.cmp(right))
        });
    }
}
