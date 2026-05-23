use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Arch {
    Avx2,
    Avx512,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    I32,
    I64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VectorState {
    top: Vec<u64>,
    bottom: Vec<u64>,
}

impl VectorState {
    pub fn new(top: Vec<u64>, bottom: Vec<u64>) -> Self {
        Self { top, bottom }
    }

    pub fn top(&self) -> &[u64] {
        &self.top
    }

    pub fn bottom(&self) -> &[u64] {
        &self.bottom
    }

    pub fn as_tuple(&self) -> (Vec<u64>, Vec<u64>) {
        (self.top.clone(), self.bottom.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum InstructionArg {
    Input(String),
    U64(u64),
    BitVec { bits: u32, hex: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstructionSpec {
    intrinsic_name: &'static str,
    args: BTreeMap<&'static str, InstructionArg>,
}

pub type InstructionSortKey = (String, Vec<(String, String)>);
pub type GadgetSortKey = (Vec<InstructionSortKey>, Vec<InstructionSortKey>);

impl InstructionSpec {
    pub fn new(intrinsic_name: &'static str, args: BTreeMap<&'static str, InstructionArg>) -> Self {
        Self {
            intrinsic_name,
            args,
        }
    }

    pub fn intrinsic_name(&self) -> &'static str {
        self.intrinsic_name
    }

    pub fn arg_u64(&self, name: &'static str) -> Option<u64> {
        match self.args.get(name) {
            Some(InstructionArg::U64(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn args(&self) -> &BTreeMap<&'static str, InstructionArg> {
        &self.args
    }

    pub fn sort_key(&self) -> InstructionSortKey {
        (
            self.intrinsic_name.to_owned(),
            self.args
                .iter()
                .map(|(key, value)| ((*key).to_owned(), instruction_arg_sort_value(value)))
                .collect(),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct InstructionSignature {
    intrinsic_name: &'static str,
    args: Vec<(&'static str, ResolvedInstructionArg)>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum ResolvedInstructionArg {
    Input(String),
    U64(u64),
    BitVec { bits: u32, hex: String },
    Instruction(Box<InstructionSignature>),
}

fn instruction_arg_sort_value(value: &InstructionArg) -> String {
    match value {
        InstructionArg::Input(name) => name.clone(),
        InstructionArg::U64(value) => value.to_string(),
        InstructionArg::BitVec { bits, hex } => format!("bits={bits},hex={hex}"),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PermutationGadget {
    top_instructions: Vec<InstructionSpec>,
    bottom_instructions: Vec<InstructionSpec>,
    validated: bool,
}

impl PermutationGadget {
    pub fn new(
        top_instructions: Vec<InstructionSpec>,
        bottom_instructions: Vec<InstructionSpec>,
    ) -> Self {
        Self {
            top_instructions,
            bottom_instructions,
            validated: true,
        }
    }

    pub fn top_instructions(&self) -> &[InstructionSpec] {
        &self.top_instructions
    }

    pub fn bottom_instructions(&self) -> &[InstructionSpec] {
        &self.bottom_instructions
    }

    pub fn validated(&self) -> bool {
        self.validated
    }

    pub fn sort_key(&self) -> GadgetSortKey {
        (
            self.top_instructions
                .iter()
                .map(InstructionSpec::sort_key)
                .collect(),
            self.bottom_instructions
                .iter()
                .map(InstructionSpec::sort_key)
                .collect(),
        )
    }

    pub fn instruction_count(&self) -> usize {
        let (unified, _, _) = self.unified_instructions();
        unified.len()
    }

    pub fn unified_instructions(&self) -> (Vec<InstructionSpec>, isize, isize) {
        let mut unified = Vec::new();
        let mut signature_to_index = BTreeMap::new();

        let top_indices = append_unified_chain(
            &self.top_instructions,
            &mut unified,
            &mut signature_to_index,
        );
        let bottom_indices = append_unified_chain(
            &self.bottom_instructions,
            &mut unified,
            &mut signature_to_index,
        );

        let top_output = top_indices.last().copied().map_or(-1, |idx| idx as isize);
        let bottom_output = bottom_indices
            .last()
            .copied()
            .map_or(-1, |idx| idx as isize);

        (unified, top_output, bottom_output)
    }
}

fn append_unified_chain(
    instructions: &[InstructionSpec],
    unified: &mut Vec<InstructionSpec>,
    signature_to_index: &mut BTreeMap<InstructionSignature, usize>,
) -> Vec<usize> {
    let mut prior_signatures = Vec::new();
    let mut indices = Vec::new();

    for instruction in instructions {
        let signature = instruction_signature(instruction, &prior_signatures);
        prior_signatures.push(signature.clone());
        let index = match signature_to_index.get(&signature) {
            Some(index) => *index,
            None => {
                let index = unified.len();
                signature_to_index.insert(signature, index);
                unified.push(instruction.clone());
                index
            }
        };
        indices.push(index);
    }

    indices
}

fn instruction_signature(
    instruction: &InstructionSpec,
    prior_signatures: &[InstructionSignature],
) -> InstructionSignature {
    InstructionSignature {
        intrinsic_name: instruction.intrinsic_name,
        args: instruction
            .args
            .iter()
            .map(|(key, value)| (*key, resolved_instruction_arg(value, prior_signatures)))
            .collect(),
    }
}

fn resolved_instruction_arg(
    value: &InstructionArg,
    prior_signatures: &[InstructionSignature],
) -> ResolvedInstructionArg {
    match value {
        InstructionArg::Input(name) if name == "prev" => prior_signatures
            .last()
            .cloned()
            .map(|signature| ResolvedInstructionArg::Instruction(Box::new(signature)))
            .unwrap_or_else(|| ResolvedInstructionArg::Input(name.clone())),
        InstructionArg::Input(name) if name.starts_with("result_") => name
            .strip_prefix("result_")
            .and_then(|suffix| suffix.parse::<usize>().ok())
            .and_then(|index| prior_signatures.get(index).cloned())
            .map(|signature| ResolvedInstructionArg::Instruction(Box::new(signature)))
            .unwrap_or_else(|| ResolvedInstructionArg::Input(name.clone())),
        InstructionArg::Input(name) => ResolvedInstructionArg::Input(name.clone()),
        InstructionArg::U64(value) => ResolvedInstructionArg::U64(*value),
        InstructionArg::BitVec { bits, hex } => ResolvedInstructionArg::BitVec {
            bits: *bits,
            hex: hex.clone(),
        },
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputRef {
    pub name: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Symbolic {
    pub name: String,
    pub bit_width: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MuxPruning {
    ForceLast,
    AtLeastOneLast,
}

impl MuxPruning {
    pub fn as_str(&self) -> &'static str {
        match self {
            MuxPruning::ForceLast => "force_last",
            MuxPruning::AtLeastOneLast => "at_least_one_last",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MuxNode {
    pub select: Symbolic,
    pub sources: Vec<GadgetNode>,
    pub pruning: Option<MuxPruning>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GadgetNode {
    Input(InputRef),
    Symbolic(Symbolic),
    Mux(MuxNode),
    Intrinsic(Box<IntrinsicNode>),
}

pub type Operand = GadgetNode;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntrinsicNode {
    pub name: &'static str,
    pub operands: Vec<(&'static str, GadgetNode)>,
    pub isomorphic_order: bool,
}

impl IntrinsicNode {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            operands: Vec::new(),
            isomorphic_order: true,
        }
    }

    pub fn with_operand(mut self, key: &'static str, operand: GadgetNode) -> Self {
        self.operands.push((key, operand));
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GadgetGraph {
    pub top: Option<IntrinsicNode>,
    pub bottom: Option<IntrinsicNode>,
}

impl GadgetGraph {
    pub fn new(top: Option<IntrinsicNode>, bottom: Option<IntrinsicNode>) -> Self {
        Self { top, bottom }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SynthesisOptions {
    pub max_unique_outputs: usize,
    pub allow_any_lane_order: bool,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_unique_outputs: 3,
            allow_any_lane_order: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SynthesisError {
    UnsupportedIntrinsic(&'static str),
    UnsupportedGadgetDepth {
        depth: u8,
        max_supported: u8,
    },
    MissingOperand {
        intrinsic: &'static str,
        operand: &'static str,
    },
    InvalidInputState {
        expected_lanes: usize,
        top: usize,
        bottom: usize,
    },
    ModelMissingValue(String),
    UnsupportedMuxEvaluation,
    UnsupportedMuxConcretization,
}

impl fmt::Display for SynthesisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SynthesisError::UnsupportedIntrinsic(name) => {
                write!(formatter, "unsupported intrinsic '{name}'")
            }
            SynthesisError::UnsupportedGadgetDepth {
                depth,
                max_supported,
            } => write!(
                formatter,
                "gadget depths above {max_supported} are not supported yet (requested {depth})"
            ),
            SynthesisError::MissingOperand { intrinsic, operand } => write!(
                formatter,
                "missing operand '{operand}' for intrinsic '{intrinsic}'"
            ),
            SynthesisError::InvalidInputState {
                expected_lanes,
                top,
                bottom,
            } => write!(
                formatter,
                "invalid input state: expected {expected_lanes} lanes, got top={top}, bottom={bottom}"
            ),
            SynthesisError::ModelMissingValue(name) => {
                write!(formatter, "model missing value for '{name}'")
            }
            SynthesisError::UnsupportedMuxEvaluation => write!(
                formatter,
                "mux evaluation is not supported by the Rust gadget synthesizer yet"
            ),
            SynthesisError::UnsupportedMuxConcretization => write!(
                formatter,
                "mux concretization is not supported by the Rust gadget synthesizer yet"
            ),
        }
    }
}

impl std::error::Error for SynthesisError {}
