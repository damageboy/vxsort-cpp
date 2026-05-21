use gadget_synth::PermutationGadget;

use crate::transition_table::{CompletePath, StateTuple, TransitionTable};

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

    fn score_assigned_path(&self, assigned_path: &AssignedPath) -> PathCost;

    fn score_path(&self, path: &CompletePath, table: &TransitionTable) -> PathCost {
        self.assign_path_gadgets(path, table)
            .map(|assigned_path| self.score_assigned_path(&assigned_path))
            .unwrap_or_else(|| PathCost::new(0, f64::INFINITY, f64::INFINITY))
    }
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
