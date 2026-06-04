use std::collections::{HashMap, HashSet};

use gadget_synth::{PermutationGadget, VectorState};

pub type LaneLabel = u8;
/// Compatibility state representation used at import/export/verifier/comment boundaries.
pub type StateTuple = (Vec<u64>, Vec<u64>);
pub type TransitionKey = (StateTuple, StateTuple);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
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

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct State {
    lanes: Vec<LaneLabel>,
}

impl State {
    pub fn from_top_bottom<const LANES: usize>(
        top: [LaneLabel; LANES],
        bottom: [LaneLabel; LANES],
    ) -> Self {
        let mut lanes = Vec::with_capacity(2 * LANES);
        lanes.extend(top);
        lanes.extend(bottom);
        Self { lanes }
    }

    pub fn try_from_zero_based_u64(top: &[u64], bottom: &[u64]) -> Result<Self, String> {
        if top.len() != bottom.len() {
            return Err(format!(
                "state top/bottom lane counts differ: {} vs {}",
                top.len(),
                bottom.len()
            ));
        }
        let mut lanes = Vec::with_capacity(top.len() + bottom.len());
        for label in top.iter().chain(bottom.iter()) {
            let label = LaneLabel::try_from(*label)
                .map_err(|_| format!("state label {label} does not fit in u8"))?;
            lanes.push(label);
        }
        Ok(Self { lanes })
    }

    pub fn try_from_one_based_u64(top: &[u64], bottom: &[u64]) -> Result<Self, String> {
        if top.len() != bottom.len() {
            return Err(format!(
                "state top/bottom lane counts differ: {} vs {}",
                top.len(),
                bottom.len()
            ));
        }
        let mut lanes = Vec::with_capacity(top.len() + bottom.len());
        for label in top.iter().chain(bottom.iter()) {
            if *label == 0 {
                return Err("one-based state label 0 is out of range".to_owned());
            }
            let zero_based = label - 1;
            let zero_based = LaneLabel::try_from(zero_based)
                .map_err(|_| format!("state label {label} does not fit in u8"))?;
            lanes.push(zero_based);
        }
        Ok(Self { lanes })
    }

    pub fn lanes(&self) -> &[LaneLabel] {
        &self.lanes
    }

    pub fn lanes_per_side(&self) -> usize {
        self.lanes.len() / 2
    }

    pub fn top(&self) -> &[LaneLabel] {
        let split = self.lanes_per_side();
        &self.lanes[..split]
    }

    pub fn bottom(&self) -> &[LaneLabel] {
        let split = self.lanes_per_side();
        &self.lanes[split..]
    }

    pub fn to_zero_based_tuple_u64(&self) -> StateTuple {
        (
            self.top().iter().map(|label| u64::from(*label)).collect(),
            self.bottom()
                .iter()
                .map(|label| u64::from(*label))
                .collect(),
        )
    }

    pub fn to_one_based_tuple_u64(&self) -> StateTuple {
        (
            self.top()
                .iter()
                .map(|label| u64::from(*label) + 1)
                .collect(),
            self.bottom()
                .iter()
                .map(|label| u64::from(*label) + 1)
                .collect(),
        )
    }

    pub fn to_zero_based_vector_state(&self) -> VectorState {
        let (top, bottom) = self.to_zero_based_tuple_u64();
        VectorState::new(top, bottom)
    }

    pub fn to_one_based_vector_state(&self) -> VectorState {
        let (top, bottom) = self.to_one_based_tuple_u64();
        VectorState::new(top, bottom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateInterner {
    lanes_per_side: Option<usize>,
    states: Vec<State>,
    by_state: HashMap<State, StateId>,
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

    pub fn intern(&mut self, state: State) -> StateId {
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

    pub fn lookup(&self, state: &State) -> Option<StateId> {
        self.by_state.get(state).copied()
    }

    pub fn get(&self, id: StateId) -> &State {
        &self.states[id.0 as usize]
    }

    fn ensure_shape(&mut self, state: &State) {
        assert!(
            state.lanes.len().is_multiple_of(2),
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
    transitions: Vec<TransitionIndex>,
}

impl CompletePath {
    pub fn new(transitions: Vec<TransitionIndex>) -> Self {
        Self { transitions }
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionRecord {
    input: StateId,
    output: StateId,
    gadgets: Vec<PermutationGadget>,
}

impl TransitionRecord {
    pub fn input(&self) -> StateId {
        self.input
    }

    pub fn output(&self) -> StateId {
        self.output
    }

    pub fn gadgets(&self) -> &[PermutationGadget] {
        &self.gadgets
    }

    pub fn gadget(&self, index: GadgetIndex) -> &PermutationGadget {
        &self.gadgets[index.0 as usize]
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionTable {
    states: StateInterner,
    stages: Vec<StageData>,
}

#[derive(Debug)]
pub struct ResolvedPathStep<'a> {
    pub stage: usize,
    pub transition: TransitionRef,
    pub input: StateId,
    pub output: StateId,
    pub input_state: StateTuple,
    pub output_state: StateTuple,
    pub gadgets: &'a [PermutationGadget],
}

impl TransitionTable {
    pub fn new(num_stages: usize) -> Self {
        Self {
            states: StateInterner::inferred(),
            stages: vec![StageData::default(); num_stages],
        }
    }

    pub fn new_with_lanes(num_stages: usize, lanes_per_side: usize) -> Self {
        Self {
            states: StateInterner::new(lanes_per_side),
            stages: vec![StageData::default(); num_stages],
        }
    }

    pub fn states(&self) -> &StateInterner {
        &self.states
    }

    pub fn state(&self, id: StateId) -> &State {
        self.states.get(id)
    }

    pub fn state_as_zero_based_tuple(&self, id: StateId) -> StateTuple {
        self.state(id).to_zero_based_tuple_u64()
    }

    pub fn state_as_one_based_tuple(&self, id: StateId) -> StateTuple {
        self.state(id).to_one_based_tuple_u64()
    }

    pub fn state_as_vector_state(&self, id: StateId) -> VectorState {
        self.state(id).to_zero_based_vector_state()
    }

    pub fn intern_zero_based_state(&mut self, state: &VectorState) -> StateId {
        let state = State::try_from_zero_based_u64(state.top(), state.bottom())
            .expect("vector state labels should fit in compact state storage");
        self.states.intern(state)
    }

    pub fn intern_zero_based_tuple(&mut self, state: &StateTuple) -> StateId {
        let state = State::try_from_zero_based_u64(&state.0, &state.1)
            .expect("state tuple labels should fit in compact state storage");
        self.states.intern(state)
    }

    pub fn intern_one_based_tuple(&mut self, state: &StateTuple) -> StateId {
        let state = State::try_from_one_based_u64(&state.0, &state.1)
            .expect("one-based state tuple labels should fit in compact state storage");
        self.states.intern(state)
    }

    pub fn lookup_zero_based_tuple(&self, state: &StateTuple) -> Option<StateId> {
        let state = State::try_from_zero_based_u64(&state.0, &state.1).ok()?;
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

    pub fn transition_gadgets(&self, reference: TransitionRef) -> &[PermutationGadget] {
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
        let input = State::try_from_one_based_u64(&input_tuple.0, &input_tuple.1).ok()?;
        let output = State::try_from_one_based_u64(&output_tuple.0, &output_tuple.1).ok()?;
        let input = self.states.lookup(&input)?;
        let output = self.states.lookup(&output)?;
        self.transition_ref_for_ids(stage, input, output)
    }

    pub fn add_transition_by_id(
        &mut self,
        stage: usize,
        input: StateId,
        output: State,
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
            gadgets: vec![gadget],
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
        let output = State::try_from_zero_based_u64(output_state.top(), output_state.bottom())
            .expect("vector state labels should fit in compact state storage");
        self.add_transition_by_id(stage, input, output, gadget)
            .gadget_was_new
    }

    pub fn add_transition_one_based_tuples(
        &mut self,
        stage: usize,
        input_tuple: &StateTuple,
        output_tuple: &StateTuple,
        gadget: PermutationGadget,
    ) -> TransitionInsertResult {
        let input = State::try_from_one_based_u64(&input_tuple.0, &input_tuple.1)
            .expect("one-based input state labels should fit in compact state storage");
        let output = State::try_from_one_based_u64(&output_tuple.0, &output_tuple.1)
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
                record.gadgets.clone(),
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
                    record.gadgets.clone(),
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
                let vector_state = VectorState::new(state.0.clone(), state.1.clone());
                (state, vector_state)
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

    pub fn resolve_complete_path(&self, path: &CompletePath) -> Vec<ResolvedPathStep<'_>> {
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
