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
