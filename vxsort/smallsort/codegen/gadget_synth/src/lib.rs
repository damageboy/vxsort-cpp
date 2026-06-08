mod intrinsics;
mod synthesizer;
mod types;

pub use intrinsics::dispatch_intrinsic;
pub use synthesizer::GadgetSynthesizer;
pub use types::{
    Arch, DType, GadgetGraph, GadgetNode, InputRef, InstructionArg, InstructionSpec, IntrinsicNode,
    LaneLabel, MuxNode, MuxPruning, Operand, OperandBinding, OperandKey, PermutationGadget,
    Symbolic, SynthesisError, SynthesisOptions, TargetPair, VectorState,
};
