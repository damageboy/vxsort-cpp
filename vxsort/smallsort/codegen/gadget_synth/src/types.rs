use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

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

/// Compact lane identity label used by synthesis state tracking.
pub type LaneLabel = u8;

/// One comparator target pair expressed as zero-based lane labels.
pub type TargetPair = (LaneLabel, LaneLabel);

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct VectorState {
    top: Box<[LaneLabel]>,
    bottom: Box<[LaneLabel]>,
}

impl VectorState {
    pub fn new(top: Vec<LaneLabel>, bottom: Vec<LaneLabel>) -> Self {
        assert_eq!(
            top.len(),
            bottom.len(),
            "vector state top/bottom lane counts differ"
        );
        Self {
            top: top.into_boxed_slice(),
            bottom: bottom.into_boxed_slice(),
        }
    }

    pub fn from_top_bottom<const LANES: usize>(
        top: [LaneLabel; LANES],
        bottom: [LaneLabel; LANES],
    ) -> Self {
        Self {
            top: top.into_iter().collect::<Vec<_>>().into_boxed_slice(),
            bottom: bottom.into_iter().collect::<Vec<_>>().into_boxed_slice(),
        }
    }

    pub fn try_from_zero_based_u64(top: &[u64], bottom: &[u64]) -> Result<Self, String> {
        if top.len() != bottom.len() {
            return Err(format!(
                "state top/bottom lane counts differ: {} vs {}",
                top.len(),
                bottom.len()
            ));
        }
        let top = top
            .iter()
            .map(|label| {
                LaneLabel::try_from(*label)
                    .map_err(|_| format!("state label {label} does not fit in u8"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bottom = bottom
            .iter()
            .map(|label| {
                LaneLabel::try_from(*label)
                    .map_err(|_| format!("state label {label} does not fit in u8"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(top, bottom))
    }

    pub fn try_from_zero_based_labels(
        top: &[LaneLabel],
        bottom: &[LaneLabel],
    ) -> Result<Self, String> {
        if top.len() != bottom.len() {
            return Err(format!(
                "state top/bottom lane counts differ: {} vs {}",
                top.len(),
                bottom.len()
            ));
        }
        Ok(Self {
            top: top.into(),
            bottom: bottom.into(),
        })
    }

    pub fn try_from_one_based_u64(top: &[u64], bottom: &[u64]) -> Result<Self, String> {
        if top.len() != bottom.len() {
            return Err(format!(
                "state top/bottom lane counts differ: {} vs {}",
                top.len(),
                bottom.len()
            ));
        }
        let mut converted_top = Vec::with_capacity(top.len());
        for label in top {
            if *label == 0 {
                return Err("one-based state label 0 is out of range".to_owned());
            }
            let zero_based = label - 1;
            converted_top.push(
                LaneLabel::try_from(zero_based)
                    .map_err(|_| format!("state label {label} does not fit in u8"))?,
            );
        }
        let mut converted_bottom = Vec::with_capacity(bottom.len());
        for label in bottom {
            if *label == 0 {
                return Err("one-based state label 0 is out of range".to_owned());
            }
            let zero_based = label - 1;
            converted_bottom.push(
                LaneLabel::try_from(zero_based)
                    .map_err(|_| format!("state label {label} does not fit in u8"))?,
            );
        }
        Ok(Self::new(converted_top, converted_bottom))
    }

    pub fn try_from_one_based_labels(
        top: &[LaneLabel],
        bottom: &[LaneLabel],
    ) -> Result<Self, String> {
        if top.len() != bottom.len() {
            return Err(format!(
                "state top/bottom lane counts differ: {} vs {}",
                top.len(),
                bottom.len()
            ));
        }
        let mut converted_top = Vec::with_capacity(top.len());
        for label in top {
            if *label == 0 {
                return Err("one-based state label 0 is out of range".to_owned());
            }
            converted_top.push(label - 1);
        }
        let mut converted_bottom = Vec::with_capacity(bottom.len());
        for label in bottom {
            if *label == 0 {
                return Err("one-based state label 0 is out of range".to_owned());
            }
            converted_bottom.push(label - 1);
        }
        Ok(Self::new(converted_top, converted_bottom))
    }

    pub fn lanes_per_side(&self) -> usize {
        self.top.len()
    }

    pub fn top(&self) -> &[LaneLabel] {
        &self.top
    }

    pub fn bottom(&self) -> &[LaneLabel] {
        &self.bottom
    }

    pub fn as_tuple(&self) -> (Vec<LaneLabel>, Vec<LaneLabel>) {
        self.to_zero_based_tuple()
    }

    pub fn to_zero_based_tuple(&self) -> (Vec<LaneLabel>, Vec<LaneLabel>) {
        (self.top().to_vec(), self.bottom().to_vec())
    }

    pub fn to_one_based_tuple(&self) -> (Vec<LaneLabel>, Vec<LaneLabel>) {
        (
            self.top()
                .iter()
                .map(|label| label.checked_add(1).expect("lane label should stay in u8"))
                .collect(),
            self.bottom()
                .iter()
                .map(|label| label.checked_add(1).expect("lane label should stay in u8"))
                .collect(),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum InstructionArg {
    Input(String),
    U64(u64),
    BitVec { bits: u32, hex: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InstructionSpec {
    intrinsic_name: String,
    args: BTreeMap<String, InstructionArg>,
}

pub type InstructionSortKey = (String, Vec<(String, String)>);
pub type GadgetSortKey = (Vec<InstructionSortKey>, Vec<InstructionSortKey>);

impl InstructionSpec {
    pub fn new(intrinsic_name: &'static str, args: BTreeMap<&'static str, InstructionArg>) -> Self {
        Self {
            intrinsic_name: intrinsic_name.to_owned(),
            args: args
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value))
                .collect(),
        }
    }

    pub fn intrinsic_name(&self) -> &str {
        &self.intrinsic_name
    }

    pub fn arg_u64(&self, name: &'static str) -> Option<u64> {
        match self.args.get(name) {
            Some(InstructionArg::U64(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn args(&self) -> &BTreeMap<String, InstructionArg> {
        &self.args
    }

    pub fn sort_key(&self) -> InstructionSortKey {
        (
            self.intrinsic_name.clone(),
            self.args
                .iter()
                .map(|(key, value)| (key.clone(), instruction_arg_sort_value(value)))
                .collect(),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct InstructionSignature {
    intrinsic_name: String,
    args: Vec<(String, ResolvedInstructionArg)>,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
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
        intrinsic_name: instruction.intrinsic_name.clone(),
        args: instruction
            .args
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    resolved_instruction_arg(value, prior_signatures),
                )
            })
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

/// Operand name accepted by supported intrinsic dispatch implementations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum OperandKey {
    /// First vector input operand.
    A,
    /// Second vector input operand or vector control operand, depending on the intrinsic.
    B,
    /// Passthrough source operand used by AVX-512 masked intrinsics.
    Src,
    /// AVX-512 mask operand.
    K,
    /// Immediate 8-bit control operand.
    Imm8,
    /// Vector index/control operand.
    OpIdx,
}

impl OperandKey {
    /// Returns the intrinsic dispatch name for this operand.
    pub fn as_str(self) -> &'static str {
        match self {
            OperandKey::A => "a",
            OperandKey::B => "b",
            OperandKey::Src => "src",
            OperandKey::K => "k",
            OperandKey::Imm8 => "imm8",
            OperandKey::OpIdx => "op_idx",
        }
    }
}

/// One named operand wired into an [`IntrinsicNode`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperandBinding {
    /// Intrinsic operand identity.
    pub key: OperandKey,

    /// Graph node that provides this operand value.
    pub node: GadgetNode,
}

impl OperandBinding {
    /// Creates a typed operand binding.
    pub fn new(key: OperandKey, node: GadgetNode) -> Self {
        Self { key, node }
    }
}

/// Symbolic template for one SIMD intrinsic in a candidate gadget.
///
/// An `IntrinsicNode` names the intrinsic to evaluate and wires each operand to
/// another [`GadgetNode`]. Operands may be input registers, symbolic controls,
/// mux-selected sources, or child intrinsic results. The synthesizer evaluates
/// this graph with Z3, then concretizes satisfying symbolic values into
/// [`InstructionSpec`] entries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntrinsicNode {
    /// Intrinsic function name understood by `intrinsics::dispatch_intrinsic`.
    pub name: &'static str,

    /// Named operands passed to the intrinsic.
    ///
    /// Keys must match the dispatch implementation for [`Self::name`]. The slice
    /// order is fixed after construction and preserved for deterministic graph
    /// keys and topological concretization, but the operand keys define the
    /// intrinsic call shape.
    pub operands: Box<[OperandBinding]>,

    /// Whether swapping the intrinsic's register operands is redundant for
    /// candidate generation.
    ///
    /// `true` lets the template generator keep one operand ordering. `false`
    /// means both register orders must be explored because they can produce
    /// distinct ordered outputs.
    pub isomorphic_order: bool,
}

impl IntrinsicNode {
    /// Creates an intrinsic template with no operands.
    ///
    /// The node defaults to [`Self::isomorphic_order`] = `true`; callers that
    /// build non-isomorphic dual-input intrinsics should set that field
    /// explicitly.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            operands: Box::new([]),
            isomorphic_order: true,
        }
    }

    /// Sets all operands for the intrinsic template at once.
    pub fn with_operands<const N: usize>(mut self, operands: [OperandBinding; N]) -> Self {
        self.operands = Box::new(operands);
        self
    }

    /// Appends a named operand and returns the updated node.
    ///
    /// This method is intended for construction-time chaining. It may reallocate
    /// while building the node, but the stored operands are boxed once the node is
    /// returned.
    pub fn with_operand(mut self, key: OperandKey, operand: GadgetNode) -> Self {
        let mut operands = self.operands.into_vec();
        operands.push(OperandBinding::new(key, operand));
        self.operands = operands.into_boxed_slice();
        self
    }

    /// Sets whether swapped register operands are redundant for this intrinsic.
    pub fn with_isomorphic_order(mut self, isomorphic_order: bool) -> Self {
        self.isomorphic_order = isomorphic_order;
        self
    }
}

/// Two-output candidate gadget evaluated by the synthesizer.
///
/// A `GadgetGraph` describes how to compute the candidate `top` and `bottom`
/// output registers from the input register pair. Each side is an optional
/// intrinsic root: `Some(node)` evaluates that intrinsic tree, while `None`
/// passes through the corresponding input register unchanged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GadgetGraph {
    /// Root intrinsic tree for the top output register, or pass-through if absent.
    pub top: Option<IntrinsicNode>,

    /// Root intrinsic tree for the bottom output register, or pass-through if absent.
    pub bottom: Option<IntrinsicNode>,
}

impl GadgetGraph {
    /// Creates a two-output gadget graph from optional top and bottom roots.
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
    UnsupportedIntrinsic(String),
    UnsupportedGadgetDepth {
        depth: u8,
        max_supported: u8,
    },
    MissingOperand {
        intrinsic: String,
        operand: String,
    },
    InvalidInputState {
        expected_lanes: usize,
        top: usize,
        bottom: usize,
    },
    ModelMissingValue(String),
    WorkerProcess(String),
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
            SynthesisError::WorkerProcess(message) => {
                write!(formatter, "worker process failed: {message}")
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
