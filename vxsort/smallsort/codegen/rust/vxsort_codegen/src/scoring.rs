use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use gadget_synth::PermutationGadget;

use crate::transition_table::{CompletePath, StateTuple, TransitionTable};

pub type AssignedPathKey = Vec<(usize, StateTuple, StateTuple, usize)>;

#[derive(Clone, Debug, PartialEq)]
pub struct GadgetCost {
    instruction_count: u32,
    latency: f64,
    throughput: f64,
    score: f64,
}

impl GadgetCost {
    pub fn new(instruction_count: u32, latency: f64, throughput: f64, score: f64) -> Self {
        Self {
            instruction_count,
            latency,
            throughput,
            score,
        }
    }

    pub fn instruction_count(&self) -> u32 {
        self.instruction_count
    }

    pub fn latency(&self) -> f64 {
        self.latency
    }

    pub fn throughput(&self) -> f64 {
        self.throughput
    }

    pub fn score(&self) -> f64 {
        self.score
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PathCost {
    instruction_count: u32,
    estimated_cycles: f64,
    score: f64,
}

impl PathCost {
    pub fn new(instruction_count: u32, estimated_cycles: f64, score: f64) -> Self {
        Self {
            instruction_count,
            estimated_cycles,
            score,
        }
    }

    pub fn instruction_count(&self) -> u32 {
        self.instruction_count
    }

    pub fn estimated_cycles(&self) -> f64 {
        self.estimated_cycles
    }

    pub fn score(&self) -> f64 {
        self.score
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssignedStep {
    stage: usize,
    input: StateTuple,
    output: StateTuple,
    gadget_index: usize,
    gadget: PermutationGadget,
}

impl AssignedStep {
    pub fn new(
        stage: usize,
        input: StateTuple,
        output: StateTuple,
        gadget_index: usize,
        gadget: PermutationGadget,
    ) -> Self {
        Self {
            stage,
            input,
            output,
            gadget_index,
            gadget,
        }
    }

    pub fn stage(&self) -> usize {
        self.stage
    }

    pub fn input(&self) -> &StateTuple {
        &self.input
    }

    pub fn output(&self) -> &StateTuple {
        &self.output
    }

    pub fn gadget_index(&self) -> usize {
        self.gadget_index
    }

    pub fn gadget(&self) -> &PermutationGadget {
        &self.gadget
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssignedPath {
    steps: Vec<AssignedStep>,
}

impl AssignedPath {
    pub fn new(steps: Vec<AssignedStep>) -> Self {
        Self { steps }
    }

    pub fn steps(&self) -> &[AssignedStep] {
        &self.steps
    }

    pub fn as_complete_path(&self) -> CompletePath {
        self.steps
            .iter()
            .map(|step| (step.stage, step.input.clone(), step.output.clone()))
            .collect()
    }

    pub fn selection_key(&self) -> AssignedPathKey {
        self.steps
            .iter()
            .map(|step| {
                (
                    step.stage,
                    step.input.clone(),
                    step.output.clone(),
                    step.gadget_index,
                )
            })
            .collect()
    }
}

pub trait Scorer {
    fn score_gadget(&self, gadget: &PermutationGadget) -> GadgetCost;

    fn assign_path_gadgets(
        &self,
        path: &CompletePath,
        table: &TransitionTable,
    ) -> Option<AssignedPath> {
        let mut steps = Vec::with_capacity(path.len());
        for (stage, input, output) in path {
            let gadgets = table
                .get_all_transitions(*stage)
                .get(&(input.clone(), output.clone()))?;
            let (gadget_index, gadget) = gadgets.iter().enumerate().min_by(|left, right| {
                self.score_gadget(left.1)
                    .score()
                    .total_cmp(&self.score_gadget(right.1).score())
                    .then_with(|| left.0.cmp(&right.0))
            })?;
            steps.push(AssignedStep::new(
                *stage,
                input.clone(),
                output.clone(),
                gadget_index,
                gadget.clone(),
            ));
        }

        Some(AssignedPath::new(steps))
    }

    fn assign_path_gadgets_k_best(
        &self,
        path: &CompletePath,
        table: &TransitionTable,
        limit: usize,
    ) -> Vec<AssignedPath> {
        if limit == 0 {
            return Vec::new();
        }
        if path.is_empty() {
            return vec![AssignedPath::new(Vec::new())];
        }

        let mut choices = Vec::with_capacity(path.len());
        for (stage, input, output) in path {
            let Some(gadgets) = table
                .get_all_transitions(*stage)
                .get(&(input.clone(), output.clone()))
            else {
                return Vec::new();
            };
            let mut scored = gadgets
                .iter()
                .enumerate()
                .map(|(gadget_index, gadget)| {
                    let cost = self.score_gadget(gadget).score();
                    (gadget_index, gadget.clone(), cost)
                })
                .collect::<Vec<_>>();
            scored.sort_by(|left, right| {
                left.2
                    .total_cmp(&right.2)
                    .then_with(|| left.0.cmp(&right.0))
            });
            if scored.is_empty() {
                return Vec::new();
            }
            choices.push(scored);
        }

        let mut heap = BinaryHeap::new();
        let mut seen = HashSet::new();
        let initial = vec![0; choices.len()];
        let initial_score = assignment_score(&choices, &initial);
        seen.insert(initial.clone());
        heap.push(KBestEntry {
            score: initial_score,
            serial: 0,
            indices: initial,
        });

        let mut next_serial = 1;
        let mut assigned = Vec::with_capacity(limit);
        while let Some(entry) = heap.pop() {
            assigned.push(build_assignment(path, &choices, &entry.indices));
            if assigned.len() >= limit {
                break;
            }

            for dimension in 0..entry.indices.len() {
                let mut next = entry.indices.clone();
                next[dimension] += 1;
                if next[dimension] >= choices[dimension].len() || !seen.insert(next.clone()) {
                    continue;
                }
                let score = assignment_score(&choices, &next);
                heap.push(KBestEntry {
                    score,
                    serial: next_serial,
                    indices: next,
                });
                next_serial += 1;
            }
        }

        assigned
    }

    fn score_assigned_path(&self, assigned_path: &AssignedPath) -> PathCost;

    fn score_path(&self, path: &CompletePath, table: &TransitionTable) -> PathCost {
        self.assign_path_gadgets(path, table)
            .map(|assigned_path| self.score_assigned_path(&assigned_path))
            .unwrap_or_else(|| PathCost::new(0, f64::INFINITY, f64::INFINITY))
    }
}

#[derive(Clone, Debug)]
struct KBestEntry {
    score: f64,
    serial: usize,
    indices: Vec<usize>,
}

impl Eq for KBestEntry {}

impl PartialEq for KBestEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal && self.serial == other.serial
    }
}

impl Ord for KBestEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| other.serial.cmp(&self.serial))
    }
}

impl PartialOrd for KBestEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn assignment_score(choices: &[Vec<(usize, PermutationGadget, f64)>], indices: &[usize]) -> f64 {
    indices
        .iter()
        .enumerate()
        .map(|(step, choice)| choices[step][*choice].2)
        .sum()
}

fn build_assignment(
    path: &CompletePath,
    choices: &[Vec<(usize, PermutationGadget, f64)>],
    indices: &[usize],
) -> AssignedPath {
    let steps = path
        .iter()
        .enumerate()
        .map(|(step_index, (stage, input, output))| {
            let (gadget_index, gadget, _) = &choices[step_index][indices[step_index]];
            AssignedStep::new(
                *stage,
                input.clone(),
                output.clone(),
                *gadget_index,
                gadget.clone(),
            )
        })
        .collect();
    AssignedPath::new(steps)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DummyScorer;

impl Scorer for DummyScorer {
    fn score_gadget(&self, gadget: &PermutationGadget) -> GadgetCost {
        let instruction_count =
            (gadget.top_instructions().len() + gadget.bottom_instructions().len()) as u32;
        GadgetCost::new(
            instruction_count,
            instruction_count as f64,
            instruction_count as f64,
            instruction_count as f64,
        )
    }

    fn score_assigned_path(&self, assigned_path: &AssignedPath) -> PathCost {
        let instruction_count = assigned_path
            .steps()
            .iter()
            .map(|step| self.score_gadget(step.gadget()).instruction_count())
            .sum();
        PathCost::new(instruction_count, 10.0, 10.0)
    }
}

pub fn transition_table_from_assigned_paths(
    num_stages: usize,
    assigned_paths: &[AssignedPath],
) -> TransitionTable {
    let mut table = TransitionTable::new(num_stages);
    for assigned_path in assigned_paths {
        for step in assigned_path.steps() {
            table.add_transition(
                step.stage(),
                &gadget_synth::VectorState::new(step.input().0.clone(), step.input().1.clone()),
                &gadget_synth::VectorState::new(step.output().0.clone(), step.output().1.clone()),
                step.gadget().clone(),
            );
        }
    }
    table
}
