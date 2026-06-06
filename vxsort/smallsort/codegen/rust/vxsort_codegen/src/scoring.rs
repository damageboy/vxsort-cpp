use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use gadget_synth::PermutationGadget;

use crate::transition_table::{
    CompletePath, GadgetIndex, GadgetListSnapshot, StateTuple, TransitionRef, TransitionTable,
};

pub type AssignedPathKey = Vec<(TransitionRef, GadgetIndex)>;

const K_BEST_TIE_EXPANSION_FACTOR: usize = 10;

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

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct AssignedPath {
    path: CompletePath,
    gadgets: Vec<GadgetIndex>,
}

impl AssignedPath {
    pub fn new(path: CompletePath, gadgets: Vec<GadgetIndex>) -> Self {
        assert_eq!(
            path.len(),
            gadgets.len(),
            "assigned path must select one gadget per stage"
        );
        Self { path, gadgets }
    }

    pub fn path(&self) -> &CompletePath {
        &self.path
    }

    pub fn gadgets(&self) -> &[GadgetIndex] {
        &self.gadgets
    }

    pub fn gadget_at_stage(&self, stage: usize) -> GadgetIndex {
        self.gadgets[stage]
    }

    pub fn as_complete_path(&self) -> CompletePath {
        self.path.clone()
    }

    pub fn selection_key(&self) -> AssignedPathKey {
        self.path
            .iter()
            .zip(self.gadgets.iter().copied())
            .map(|((stage, transition), gadget)| {
                (
                    TransitionRef {
                        stage: stage
                            .try_into()
                            .expect("stage index should fit in transition ref"),
                        transition,
                    },
                    gadget,
                )
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PathScoringStep {
    stage: usize,
    transition: TransitionRef,
    input: StateTuple,
    output: StateTuple,
    gadgets: GadgetListSnapshot,
}

impl PathScoringStep {
    pub fn new(
        stage: usize,
        transition: TransitionRef,
        input: StateTuple,
        output: StateTuple,
        gadgets: GadgetListSnapshot,
    ) -> Self {
        Self {
            stage,
            transition,
            input,
            output,
            gadgets,
        }
    }

    pub fn stage(&self) -> usize {
        self.stage
    }

    pub fn transition(&self) -> TransitionRef {
        self.transition
    }

    pub fn input(&self) -> &StateTuple {
        &self.input
    }

    pub fn output(&self) -> &StateTuple {
        &self.output
    }

    pub fn gadgets(&self) -> &GadgetListSnapshot {
        &self.gadgets
    }

    pub fn gadget(&self, index: GadgetIndex) -> &PermutationGadget {
        self.gadgets.get(index.0 as usize)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PathScoringSnapshot {
    path: CompletePath,
    steps: Vec<PathScoringStep>,
}

impl PathScoringSnapshot {
    pub fn new(path: CompletePath, steps: Vec<PathScoringStep>) -> Self {
        Self { path, steps }
    }

    pub fn from_table(path: &CompletePath, table: &TransitionTable) -> Self {
        let steps = path
            .iter()
            .map(|(stage, transition)| {
                let transition_ref = TransitionRef {
                    stage: stage
                        .try_into()
                        .expect("stage index should fit in transition ref"),
                    transition,
                };
                let record = table.transition(transition_ref);
                PathScoringStep::new(
                    stage,
                    transition_ref,
                    table.state_as_zero_based_tuple(record.input()),
                    table.state_as_zero_based_tuple(record.output()),
                    record.gadgets(),
                )
            })
            .collect();
        Self::new(path.clone(), steps)
    }

    pub fn path(&self) -> &CompletePath {
        &self.path
    }

    pub fn steps(&self) -> &[PathScoringStep] {
        &self.steps
    }
}

pub trait Scorer: Send + Sync {
    fn score_gadget(&self, gadget: &PermutationGadget) -> GadgetCost;

    fn assign_path_gadgets(
        &self,
        path: &CompletePath,
        table: &TransitionTable,
    ) -> Option<AssignedPath> {
        let mut gadgets = Vec::with_capacity(path.len());
        for (stage, transition) in path.iter() {
            let transition_ref = TransitionRef {
                stage: stage
                    .try_into()
                    .expect("stage index should fit in transition ref"),
                transition,
            };
            let selected = table
                .transition_gadgets(transition_ref)
                .iter()
                .enumerate()
                .min_by(|left, right| {
                    self.score_gadget(left.1)
                        .score()
                        .total_cmp(&self.score_gadget(right.1).score())
                        .then_with(|| left.0.cmp(&right.0))
                })?
                .0;
            gadgets.push(GadgetIndex(
                selected.try_into().expect("gadget index should fit in u16"),
            ));
        }

        Some(AssignedPath::new(path.clone(), gadgets))
    }

    fn assign_path_gadgets_k_best(
        &self,
        path: &CompletePath,
        table: &TransitionTable,
        limit: usize,
    ) -> Vec<AssignedPath> {
        let snapshot = PathScoringSnapshot::from_table(path, table);
        self.assign_path_gadgets_k_best_snapshot(&snapshot, limit)
    }

    fn assign_path_gadgets_k_best_snapshot(
        &self,
        snapshot: &PathScoringSnapshot,
        limit: usize,
    ) -> Vec<AssignedPath> {
        if limit == 0 {
            return Vec::new();
        }
        if snapshot.path().is_empty() {
            return vec![AssignedPath::new(CompletePath::new(Vec::new()), Vec::new())];
        }

        let mut choices = Vec::with_capacity(snapshot.path().len());
        for step in snapshot.steps() {
            let mut scored = step
                .gadgets()
                .iter()
                .enumerate()
                .map(|(gadget_index, gadget)| {
                    let cost = self.score_gadget(gadget).score();
                    (
                        GadgetIndex(
                            gadget_index
                                .try_into()
                                .expect("gadget index should fit in u16"),
                        ),
                        cost,
                    )
                })
                .collect::<Vec<_>>();
            scored.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
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
        let max_assigned = limit.saturating_mul(K_BEST_TIE_EXPANSION_FACTOR).max(limit);
        let mut cutoff_score = None;
        let mut assigned = Vec::with_capacity(limit);
        while let Some(entry) = heap.pop() {
            if let Some(cutoff_score) = cutoff_score
                && entry.score.total_cmp(&cutoff_score).is_gt()
            {
                break;
            }
            assigned.push(build_assignment(snapshot.path(), &choices, &entry.indices));

            if assigned.len() >= limit && cutoff_score.is_none() {
                cutoff_score = Some(entry.score);
            }
            if assigned.len() >= max_assigned {
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

    fn score_assigned_path(
        &self,
        assigned_path: &AssignedPath,
        table: &TransitionTable,
    ) -> PathCost;

    fn score_assigned_path_snapshot(
        &self,
        assigned_path: &AssignedPath,
        snapshot: &PathScoringSnapshot,
    ) -> PathCost {
        let instruction_count = snapshot
            .steps()
            .iter()
            .zip(assigned_path.gadgets())
            .map(|(step, gadget_index)| {
                self.score_gadget(step.gadget(*gadget_index))
                    .instruction_count()
            })
            .sum();
        let score = snapshot
            .steps()
            .iter()
            .zip(assigned_path.gadgets())
            .map(|(step, gadget_index)| self.score_gadget(step.gadget(*gadget_index)).score())
            .sum();
        PathCost::new(instruction_count, score, score)
    }

    fn score_path(&self, path: &CompletePath, table: &TransitionTable) -> PathCost {
        self.assign_path_gadgets(path, table)
            .map(|assigned_path| self.score_assigned_path(&assigned_path, table))
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

fn assignment_score(choices: &[Vec<(GadgetIndex, f64)>], indices: &[usize]) -> f64 {
    indices
        .iter()
        .enumerate()
        .map(|(step, choice)| choices[step][*choice].1)
        .sum()
}

fn build_assignment(
    path: &CompletePath,
    choices: &[Vec<(GadgetIndex, f64)>],
    indices: &[usize],
) -> AssignedPath {
    let gadgets = indices
        .iter()
        .enumerate()
        .map(|(step, choice)| choices[step][*choice].0)
        .collect();
    AssignedPath::new(path.clone(), gadgets)
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

    fn score_assigned_path(
        &self,
        assigned_path: &AssignedPath,
        table: &TransitionTable,
    ) -> PathCost {
        let instruction_count = assigned_path
            .path()
            .iter()
            .zip(assigned_path.gadgets())
            .map(|((stage, transition), gadget_index)| {
                let transition_ref = TransitionRef {
                    stage: stage
                        .try_into()
                        .expect("stage index should fit in transition ref"),
                    transition,
                };
                self.score_gadget(table.transition(transition_ref).gadget(*gadget_index))
                    .instruction_count()
            })
            .sum();
        PathCost::new(instruction_count, 10.0, 10.0)
    }

    fn score_assigned_path_snapshot(
        &self,
        assigned_path: &AssignedPath,
        snapshot: &PathScoringSnapshot,
    ) -> PathCost {
        let instruction_count = snapshot
            .steps()
            .iter()
            .zip(assigned_path.gadgets())
            .map(|(step, gadget_index)| {
                self.score_gadget(step.gadget(*gadget_index))
                    .instruction_count()
            })
            .sum();
        PathCost::new(instruction_count, 10.0, 10.0)
    }
}
