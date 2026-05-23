//! Modeled instruction stream types for bitonic sort lowering.
//!
//! [`InstructionStream`] is the canonical intermediate representation (IR)
//! between solution-path lowering and output rendering.  Renderers – the ASM
//! emitter, a future UICA scorer, etc. – consume this type rather than
//! operating directly on [`crate::transition_table::CompletePath`].
//!
//! # Task 1 – Data model types
//!
//! The public data types ([`InstructionStream`], [`InstructionBlock`],
//! [`ModeledInstruction`], [`Operand`], [`Register`], [`ConstantData`],
//! [`ConstantKey`], [`LoweringOptions`]) are defined in this module.
//!
//! # Task 2 – Lowering entry point
//!
//! [`lower_solution_paths`] converts a [`crate::transition_table::CompletePath`]
//! list (together with a [`crate::transition_table::TransitionTable`] and
//! [`crate::json_exporter::SolutionJsonMetadata`]) into an
//! [`InstructionStream`].  Register allocation is owned by the lowering pass;
//! callers need not track register names.

use std::collections::BTreeMap;
use std::fmt;

use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget};

use crate::json_exporter::SolutionJsonMetadata;
use crate::transition_table::{CompletePath, StateTuple, TransitionKey, TransitionTable};
use crate::{ArchArg, DTypeArg};

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

/// A physical vector register used in AVX2 or AVX-512 instructions.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Register {
    /// 256-bit YMM register (AVX2), numbered 0–15.
    Ymm(u8),
    /// 512-bit ZMM register (AVX-512), numbered 0–31.
    Zmm(u8),
}

impl Register {
    /// Returns the canonical Intel/NASM assembly name, e.g. `"ymm0"` or
    /// `"zmm3"`.
    pub fn name(&self) -> String {
        match self {
            Register::Ymm(n) => format!("ymm{n}"),
            Register::Zmm(n) => format!("zmm{n}"),
        }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// Constant pool
// ---------------------------------------------------------------------------

/// Semantic key that identifies a constant-pool entry.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum ConstantKey {
    /// A vector/bit-vector payload, represented by its bit width and stable
    /// hexadecimal value string.
    Vector { bits: u32, hex: String },
    /// An AVX-512 opmask constant.
    KMask { bits: u32, value: u64 },
    /// A packed vector control operand rendered as lane elements.
    ControlVector {
        bits: u32,
        element_bits: u32,
        value: u64,
    },
}

/// One constant value owned by the stream's constant pool.
///
/// Instructions that reference rodata carry an [`Operand::ConstantRef`]
/// pointing to `label`.  `key` records the semantic value used for deterministic
/// GVN/reuse.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ConstantData {
    /// Deterministic label, e.g. `LC0`.
    pub label: String,
    /// Semantic constant key.
    pub key: ConstantKey,
}

// ---------------------------------------------------------------------------
// Operand
// ---------------------------------------------------------------------------

/// A single instruction operand in the modeled stream.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Operand {
    /// Physical vector register (YMM or ZMM).
    Register(Register),
    /// Small integer immediate.
    Immediate(u64),
    /// Reference into the stream's constant pool by label.
    ConstantRef(String),
    /// AVX-512 opmask register `k0`–`k7`, identified by index.
    KMask(u8),
    /// A vector register with AVX-512 writemask syntax, e.g. `zmm0{k1}`.
    MaskedRegister(Register, u8),
    /// A general-purpose register used for materialization.
    Gpr(&'static str),
    /// An operand that could not be lowered to a known form.
    ///
    /// Kept explicit so unsupported gadgets remain visible rather than
    /// silently disappearing from output.
    Unsupported(String),
}

// ---------------------------------------------------------------------------
// ModeledInstruction
// ---------------------------------------------------------------------------

/// One modeled instruction with operands and an optional annotation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ModeledInstruction {
    /// Assembly mnemonic, e.g. `"vpermq"`, `"vpermd"`, `"vpminsd"`.
    pub mnemonic: String,
    /// Ordered list of operands in Intel syntax (destination first).
    pub operands: Vec<Operand>,
    /// Optional human-readable comment or stage annotation.
    pub comment: Option<String>,
}

// ---------------------------------------------------------------------------
// InstructionBlock
// ---------------------------------------------------------------------------

/// A (optionally labeled) block of modeled instructions.
///
/// Corresponds to one NASM-style function label and its body.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct InstructionBlock {
    /// Optional NASM/Intel-syntax label for this block, e.g.
    /// `"vxsort_solution_0"`.
    pub label: Option<String>,
    /// Ordered list of instructions in this block.
    pub instructions: Vec<ModeledInstruction>,
}

// ---------------------------------------------------------------------------
// LoweringOptions
// ---------------------------------------------------------------------------

/// Options that influence how a solution path is lowered to an
/// [`InstructionStream`].
///
/// Skeletal for Task 1; will grow as lowering is implemented in later tasks.
#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct LoweringOptions {}

// ---------------------------------------------------------------------------
// InstructionStream
// ---------------------------------------------------------------------------

/// The canonical IR for a lowered bitonic sort solution.
///
/// Contains:
/// - a **constant pool** (`constants`) of rodata entries referenced by
///   [`Operand::ConstantRef`], and
/// - a list of **labeled instruction blocks** (`blocks`).
///
/// Renderers (ASM emitter, UICA scorer) consume this type.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct InstructionStream {
    /// Constant pool – rodata entries shared across blocks.
    pub constants: Vec<ConstantData>,
    /// Ordered list of labeled instruction blocks.
    pub blocks: Vec<InstructionBlock>,
}

// ---------------------------------------------------------------------------
// Public lowering entry point (Task 2)
// ---------------------------------------------------------------------------

/// Lower a list of complete solution paths into a canonical
/// [`InstructionStream`].
///
/// Each path in `paths` produces one [`InstructionBlock`] with label
/// `vxsort_solution_N` (where N is the zero-based path index).  Register
/// allocation is performed internally; callers need not track register names.
///
/// `_options` is reserved for future use and may be passed as
/// [`LoweringOptions::default()`].
pub fn lower_solution_paths(
    metadata: &SolutionJsonMetadata,
    table: &TransitionTable,
    paths: &[CompletePath],
    _options: LoweringOptions,
) -> InstructionStream {
    let mut constants = ConstantPool::default();
    let mut blocks = vec![];
    for (solution_index, path) in paths.iter().enumerate() {
        blocks.push(lower_path(
            solution_index,
            metadata,
            table,
            path,
            &mut constants,
        ));
    }
    InstructionStream {
        constants: constants.into_constants(),
        blocks,
    }
}

// ---------------------------------------------------------------------------
// Private lowering helpers
// ---------------------------------------------------------------------------

fn lower_path(
    solution_index: usize,
    metadata: &SolutionJsonMetadata,
    table: &TransitionTable,
    path: &CompletePath,
    constants: &mut ConstantPool,
) -> InstructionBlock {
    let mut regs = LoweringRegAlloc::new(metadata.arch, metadata.num_vecs);
    let mut materialized = MaterializedConstants::default();
    let mut instructions: Vec<ModeledInstruction> = vec![];

    if let Some((_, input, _)) = path.first() {
        push_comment(&mut instructions, "Initial Input State:");
        push_state_comments(&mut instructions, input);
    }

    for (step_index, (stage, input, output)) in path.iter().enumerate() {
        let src_top = regs.src_top();
        let src_bottom = regs.src_bottom();
        let dst_top = regs.dst_top();
        let dst_bottom = regs.dst_bottom();
        let is_final_natural_order_stage = metadata.natural_order && step_index + 1 == path.len();
        let has_coex = !is_final_natural_order_stage && metadata.num_vecs > 1;
        push_comment(&mut instructions, format!("Stage {stage}"));

        let transition_key: TransitionKey = (input.clone(), output.clone());
        let is_dual_side = if let Some(gadget) = table
            .get_all_transitions(*stage)
            .get(&transition_key)
            .and_then(|gadgets| gadgets.first())
        {
            let mut state = LoweringState {
                instructions: &mut instructions,
                regs: &mut regs,
                materialized: &mut materialized,
                constants,
            };
            lower_gadget(&mut state, gadget, *stage)
        } else {
            instructions.push(ModeledInstruction {
                mnemonic: "nop".to_owned(),
                operands: vec![Operand::Unsupported(format!(
                    "missing gadget for stage {stage}"
                ))],
                comment: Some(format!("stage {stage}: missing gadget")),
            });
            false
        };

        let output_in_dst = is_dual_side != has_coex;
        let (out_top, out_bottom) = if output_in_dst {
            (&dst_top, &dst_bottom)
        } else {
            (&src_top, &src_bottom)
        };
        push_comment(
            &mut instructions,
            format!(
                "Registers: {} = top, {} = bottom  ->  {} = top, {} = bottom",
                src_top.name(),
                src_bottom.name(),
                out_top.name(),
                out_bottom.name()
            ),
        );

        if has_coex {
            push_comment(
                &mut instructions,
                format!(
                    "Compare-swap: min -> {} (top), max -> {} (bottom)",
                    regs.dst_top().name(),
                    regs.dst_bottom().name()
                ),
            );
            lower_compare_swap(&mut instructions, &mut regs, metadata);
        }
        push_comment(&mut instructions, "Output State:");
        push_state_comments(&mut instructions, output);
        regs.reset_temps();
    }

    instructions.push(ModeledInstruction {
        mnemonic: "ret".to_owned(),
        operands: vec![],
        comment: None,
    });

    InstructionBlock {
        label: Some(format!("vxsort_solution_{solution_index}")),
        instructions,
    }
}

fn push_comment(instructions: &mut Vec<ModeledInstruction>, comment: impl Into<String>) {
    instructions.push(ModeledInstruction {
        mnemonic: String::new(),
        operands: vec![],
        comment: Some(comment.into()),
    });
}

fn push_state_comments(instructions: &mut Vec<ModeledInstruction>, state: &StateTuple) {
    push_comment(instructions, format!("top: {:?}", state.0));
    push_comment(instructions, format!("bottom: {:?}", state.1));
}

struct LoweringState<'a> {
    instructions: &'a mut Vec<ModeledInstruction>,
    regs: &'a mut LoweringRegAlloc,
    materialized: &'a mut MaterializedConstants,
    constants: &'a mut ConstantPool,
}

fn lower_gadget(state: &mut LoweringState<'_>, gadget: &PermutationGadget, stage: usize) -> bool {
    let dual_side =
        !gadget.top_instructions().is_empty() && !gadget.bottom_instructions().is_empty();
    if dual_side {
        let prefix_len = shared_prefix_len(gadget.top_instructions(), gadget.bottom_instructions());
        let mut shared_results = BTreeMap::new();
        if prefix_len > 0 {
            push_comment(
                state.instructions,
                format!(
                    "Shared instructions ({prefix_len} of {}):",
                    gadget.top_instructions().len()
                ),
            );
            shared_results = lower_instruction_chain_inner(
                state,
                &gadget.top_instructions()[..prefix_len],
                ChainLoweringContext {
                    side: LoweringSide::Top,
                    write_alternate_pair: false,
                    stage,
                    mode: ChainLoweringMode::Shared,
                    skip_prefix: 0,
                    precomputed_results: &BTreeMap::new(),
                },
            );
        }

        if gadget.top_instructions().len() > prefix_len {
            push_comment(
                state.instructions,
                format!("Top vector ({}) operations:", state.regs.src_top().name()),
            );
            lower_instruction_chain_inner(
                state,
                gadget.top_instructions(),
                ChainLoweringContext {
                    side: LoweringSide::Top,
                    write_alternate_pair: true,
                    stage,
                    mode: ChainLoweringMode::Normal,
                    skip_prefix: prefix_len,
                    precomputed_results: &shared_results,
                },
            );
        }

        if gadget.bottom_instructions().len() > prefix_len {
            push_comment(
                state.instructions,
                format!(
                    "Bottom vector ({}) operations:",
                    state.regs.src_bottom().name()
                ),
            );
            lower_instruction_chain_inner(
                state,
                gadget.bottom_instructions(),
                ChainLoweringContext {
                    side: LoweringSide::Bottom,
                    write_alternate_pair: true,
                    stage,
                    mode: ChainLoweringMode::Normal,
                    skip_prefix: prefix_len,
                    precomputed_results: &shared_results,
                },
            );
        }

        if prefix_len > 0
            && let Some(last_shared) = shared_results.get(&(prefix_len - 1)).cloned()
        {
            let mov = vector_move_mnemonic(state.regs.arch).to_owned();
            if prefix_len == gadget.top_instructions().len() {
                state.instructions.push(ModeledInstruction {
                    mnemonic: mov.clone(),
                    operands: vec![
                        Operand::Register(state.regs.dst_top()),
                        Operand::Register(last_shared.clone()),
                    ],
                    comment: Some("shared -> top".to_owned()),
                });
            }
            if prefix_len == gadget.bottom_instructions().len() {
                state.instructions.push(ModeledInstruction {
                    mnemonic: mov,
                    operands: vec![
                        Operand::Register(state.regs.dst_bottom()),
                        Operand::Register(last_shared),
                    ],
                    comment: Some("shared -> bottom".to_owned()),
                });
            }
        }

        state.regs.swap_pairs();
        true
    } else {
        lower_instruction_chain_inner(
            state,
            gadget.top_instructions(),
            ChainLoweringContext {
                side: LoweringSide::Top,
                write_alternate_pair: false,
                stage,
                mode: ChainLoweringMode::Normal,
                skip_prefix: 0,
                precomputed_results: &BTreeMap::new(),
            },
        );
        lower_instruction_chain_inner(
            state,
            gadget.bottom_instructions(),
            ChainLoweringContext {
                side: LoweringSide::Bottom,
                write_alternate_pair: false,
                stage,
                mode: ChainLoweringMode::Normal,
                skip_prefix: 0,
                precomputed_results: &BTreeMap::new(),
            },
        );
        false
    }
}

fn shared_prefix_len(top: &[InstructionSpec], bottom: &[InstructionSpec]) -> usize {
    top.iter()
        .zip(bottom)
        .take_while(|(left, right)| left == right)
        .count()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ChainLoweringMode {
    Normal,
    Shared,
}

struct ChainLoweringContext<'a> {
    side: LoweringSide,
    write_alternate_pair: bool,
    stage: usize,
    mode: ChainLoweringMode,
    skip_prefix: usize,
    precomputed_results: &'a BTreeMap<usize, Register>,
}

fn lower_instruction_chain_inner(
    state: &mut LoweringState<'_>,
    specs: &[InstructionSpec],
    context: ChainLoweringContext<'_>,
) -> BTreeMap<usize, Register> {
    if specs.is_empty() {
        return BTreeMap::new();
    }

    let source_reg = state.regs.source_for_side(context.side);
    let final_dest_reg = if context.write_alternate_pair {
        state.regs.destination_for_side(context.side)
    } else {
        source_reg.clone()
    };
    let mut previous = if context.skip_prefix > 0 {
        context
            .precomputed_results
            .get(&(context.skip_prefix - 1))
            .cloned()
            .unwrap_or_else(|| source_reg.clone())
    } else {
        source_reg.clone()
    };
    let mut result_regs: BTreeMap<usize, Register> = context.precomputed_results.clone();
    let mut emitted_results = BTreeMap::new();

    for (index, spec) in specs.iter().enumerate() {
        if index < context.skip_prefix {
            continue;
        }
        let dest = if context.mode == ChainLoweringMode::Shared {
            state.regs.allocate_temp()
        } else if index + 1 == specs.len() {
            final_dest_reg.clone()
        } else {
            state.regs.allocate_temp()
        };
        let (operands, comment_parts) =
            lower_instruction_operands(state, spec, &dest, &previous, &result_regs, context.side);
        let comment = if comment_parts.is_empty() {
            format!("stage {}", context.stage)
        } else {
            format!("stage {}; {}", context.stage, comment_parts.join(", "))
        };
        state.instructions.push(ModeledInstruction {
            mnemonic: intrinsic_to_asm_mnemonic(spec.intrinsic_name()).to_owned(),
            operands,
            comment: Some(comment),
        });
        previous = dest.clone();
        result_regs.insert(index, dest);
        emitted_results.insert(index, previous.clone());
    }
    emitted_results
}

fn lower_instruction_operands(
    state: &mut LoweringState<'_>,
    spec: &InstructionSpec,
    dest: &Register,
    previous: &Register,
    result_regs: &BTreeMap<usize, Register>,
    side: LoweringSide,
) -> (Vec<Operand>, Vec<String>) {
    let args = spec.args();
    let (a_key, a) = if let Some(arg) = args.get("a") {
        ("a", Some(arg))
    } else {
        ("src", args.get("src"))
    };
    let b = args.get("b");
    let imm = args.get("imm8").and_then(instruction_arg_u64);
    let mask = args.get("k").and_then(instruction_arg_u64);

    // No primary `a`/`src` arg means the instruction shape is unsupported.
    let Some(a_arg) = a else {
        return (
            vec![Operand::Unsupported(format!(
                "unsupported operands for {}",
                spec.intrinsic_name()
            ))],
            Vec::new(),
        );
    };

    let mut consumed_args = vec![a_key];
    let mut comment_parts = Vec::new();
    let destination =
        if let Some(mask) = mask.filter(|_| is_masked_intrinsic(spec.intrinsic_name())) {
            let register =
                state
                    .materialized
                    .materialize_kmask(state.instructions, state.constants, 16, mask);
            consumed_args.push("k");
            comment_parts.push(format!("k=0x{mask:02x}"));
            Operand::MaskedRegister(dest.clone(), register)
        } else {
            Operand::Register(dest.clone())
        };
    let mut operands = vec![destination];
    operands.push(lower_source_arg(state, a_arg, previous, result_regs, side));
    if let Some(b_arg) = b {
        consumed_args.push("b");
        if let Some(control_value) = instruction_arg_u64(b_arg)
            && intrinsic_uses_vector_control(spec.intrinsic_name())
        {
            let register_bits = register_bits_for_arch(state.regs.arch);
            let element_bits = control_vector_element_bits(spec.intrinsic_name());
            let register = state.materialized.materialize_control_vector(
                state.instructions,
                state.constants,
                state.regs,
                register_bits,
                element_bits,
                control_value,
            );
            operands.push(Operand::Register(register));
            comment_parts.push(format_control_vector(
                control_value,
                register_bits,
                element_bits,
            ));
        } else {
            operands.push(lower_source_arg(state, b_arg, previous, result_regs, side));
        }
    }
    if let Some(op_idx) = args.get("op_idx").and_then(instruction_arg_u64) {
        consumed_args.push("op_idx");
        let register_bits = register_bits_for_arch(state.regs.arch);
        let element_bits = control_vector_element_bits(spec.intrinsic_name());
        let register = state.materialized.materialize_control_vector(
            state.instructions,
            state.constants,
            state.regs,
            register_bits,
            element_bits,
            op_idx,
        );
        operands.insert(1, Operand::Register(register));
        comment_parts.push(format_control_vector(op_idx, register_bits, element_bits));
    }
    if let Some(imm_val) = imm {
        consumed_args.push("imm8");
        operands.push(Operand::Immediate(imm_val));
        if let Some(comment) = immediate_comment(spec.intrinsic_name(), imm_val) {
            comment_parts.push(comment);
        }
    }
    for (name, value) in args {
        if !consumed_args.contains(name) {
            lower_unconsumed_arg(state, &mut operands, name, value);
        }
    }
    (operands, comment_parts)
}

fn is_masked_intrinsic(name: &str) -> bool {
    name.contains("_mask_")
}

fn intrinsic_uses_vector_control(name: &str) -> bool {
    matches!(
        name,
        "_mm256_permutexvar_epi32"
            | "_mm256_permutexvar_epi64"
            | "_mm512_permutexvar_epi32"
            | "_mm512_permutexvar_epi64"
            | "_mm512_permutex2var_epi32"
            | "_mm512_permutex2var_epi64"
            | "_mm512_mask_permutex2var_epi32"
            | "_mm512_mask_permutex2var_epi64"
    )
}

fn control_vector_element_bits(name: &str) -> u32 {
    if name.contains("epi64") || name.contains("_pd") || name.contains("alignr_epi64") {
        64
    } else {
        32
    }
}

fn register_bits_for_arch(arch: ArchArg) -> u32 {
    match arch {
        ArchArg::Avx2 => 256,
        ArchArg::Avx512 => 512,
    }
}

fn format_control_vector(value: u64, bits: u32, element_bits: u32) -> String {
    format!(
        "control={}",
        format_lane_list(&control_vector_elements(value, bits, element_bits))
    )
}

fn immediate_comment(name: &str, value: u64) -> Option<String> {
    if name.contains("permute")
        || name.contains("shuffle")
        || name.contains("blend")
        || name.contains("alignr")
    {
        Some(format!("imm8=0b{value:08b}"))
    } else {
        None
    }
}

pub(crate) fn control_vector_elements(value: u64, bits: u32, element_bits: u32) -> Vec<u64> {
    let lanes = (bits / element_bits).max(1);
    (0..lanes)
        .map(|lane| (value >> (lane * 8)) & 0xff)
        .collect()
}

fn format_lane_list(elements: &[u64]) -> String {
    let rendered = elements
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{rendered}]")
}

fn lower_source_arg(
    state: &mut LoweringState<'_>,
    arg: &InstructionArg,
    previous: &Register,
    result_regs: &BTreeMap<usize, Register>,
    side: LoweringSide,
) -> Operand {
    match arg {
        InstructionArg::Input(name) if name == "top" => Operand::Register(state.regs.src_top()),
        InstructionArg::Input(name) if name == "bottom" => {
            Operand::Register(state.regs.src_bottom())
        }
        InstructionArg::Input(name) if name == "prev" => Operand::Register(previous.clone()),
        InstructionArg::Input(name) if name == "input" => {
            Operand::Register(state.regs.source_for_side(side))
        }
        InstructionArg::Input(name) if name.starts_with("result_") => name
            .strip_prefix("result_")
            .and_then(|s| s.parse::<usize>().ok())
            .and_then(|i| result_regs.get(&i).cloned())
            .map(Operand::Register)
            .unwrap_or_else(|| Operand::Unsupported(format!("unresolved {name}"))),
        InstructionArg::Input(name) => Operand::Unsupported(format!("input:{name}")),
        InstructionArg::U64(v) => Operand::Immediate(*v),
        InstructionArg::BitVec { bits, hex } => {
            Operand::Register(state.materialized.materialize_vector(
                state.instructions,
                state.constants,
                state.regs,
                *bits,
                hex,
            ))
        }
    }
}

fn lower_unconsumed_arg(
    state: &mut LoweringState<'_>,
    operands: &mut Vec<Operand>,
    name: &str,
    value: &InstructionArg,
) {
    match value {
        InstructionArg::BitVec { bits, hex } => {
            let register = state.materialized.materialize_vector(
                state.instructions,
                state.constants,
                state.regs,
                *bits,
                hex,
            );
            operands.push(Operand::Register(register));
            operands.push(Operand::Unsupported(format!(
                "unconsumed operand {name}=bitvec{bits}:{}",
                state.constants.label_for_key(&ConstantKey::Vector {
                    bits: *bits,
                    hex: hex.clone(),
                })
            )));
        }
        InstructionArg::U64(value) if name == "k" => {
            let register = state.materialized.materialize_kmask(
                state.instructions,
                state.constants,
                16,
                *value,
            );
            operands.push(Operand::KMask(register));
            operands.push(Operand::Unsupported(format!(
                "unconsumed operand {name}=kmask16:{}",
                state.constants.label_for_key(&ConstantKey::KMask {
                    bits: 16,
                    value: *value,
                })
            )));
        }
        _ => operands.push(Operand::Unsupported(format!(
            "unconsumed operand {name}={value:?}"
        ))),
    }
}

fn instruction_arg_u64(arg: &InstructionArg) -> Option<u64> {
    match arg {
        InstructionArg::U64(v) => Some(*v),
        _ => None,
    }
}

fn lower_compare_swap(
    instructions: &mut Vec<ModeledInstruction>,
    regs: &mut LoweringRegAlloc,
    metadata: &SolutionJsonMetadata,
) {
    match (metadata.arch, metadata.dtype) {
        (ArchArg::Avx2, DTypeArg::I64) => {
            let tmp = regs.allocate_temp();
            instructions.push(ModeledInstruction {
                mnemonic: "vpcmpgtq".to_owned(),
                operands: vec![
                    Operand::Register(tmp.clone()),
                    Operand::Register(regs.src_top()),
                    Operand::Register(regs.src_bottom()),
                ],
                comment: None,
            });
            instructions.push(ModeledInstruction {
                mnemonic: "vblendvpd".to_owned(),
                operands: vec![
                    Operand::Register(regs.dst_top()),
                    Operand::Register(regs.src_bottom()),
                    Operand::Register(regs.src_top()),
                    Operand::Register(tmp.clone()),
                ],
                comment: None,
            });
            instructions.push(ModeledInstruction {
                mnemonic: "vblendvpd".to_owned(),
                operands: vec![
                    Operand::Register(regs.dst_bottom()),
                    Operand::Register(regs.src_top()),
                    Operand::Register(regs.src_bottom()),
                    Operand::Register(tmp),
                ],
                comment: None,
            });
        }
        (_, DTypeArg::I32) => {
            instructions.push(ModeledInstruction {
                mnemonic: "vpminsd".to_owned(),
                operands: vec![
                    Operand::Register(regs.dst_top()),
                    Operand::Register(regs.src_top()),
                    Operand::Register(regs.src_bottom()),
                ],
                comment: None,
            });
            instructions.push(ModeledInstruction {
                mnemonic: "vpmaxsd".to_owned(),
                operands: vec![
                    Operand::Register(regs.dst_bottom()),
                    Operand::Register(regs.src_top()),
                    Operand::Register(regs.src_bottom()),
                ],
                comment: None,
            });
        }
        (_, DTypeArg::I64) => {
            instructions.push(ModeledInstruction {
                mnemonic: "vpminsq".to_owned(),
                operands: vec![
                    Operand::Register(regs.dst_top()),
                    Operand::Register(regs.src_top()),
                    Operand::Register(regs.src_bottom()),
                ],
                comment: None,
            });
            instructions.push(ModeledInstruction {
                mnemonic: "vpmaxsq".to_owned(),
                operands: vec![
                    Operand::Register(regs.dst_bottom()),
                    Operand::Register(regs.src_top()),
                    Operand::Register(regs.src_bottom()),
                ],
                comment: None,
            });
        }
    }
    regs.swap_pairs();
}

/// Map an intrinsic name to its Intel/NASM assembly mnemonic.
///
/// Mnemonic selection belongs to lowering so ASM rendering and future UICA
/// scoring consume the same modeled instruction stream.
fn intrinsic_to_asm_mnemonic(name: &str) -> &'static str {
    match name {
        "_mm256_permute4x64_epi64" | "_mm256_permutexvar_epi64" | "_mm512_permutexvar_epi64" => {
            "vpermq"
        }
        "_mm256_permute_pd" | "_mm512_permute_pd" | "_mm512_mask_permute_pd" => "vpermilpd",
        "_mm256_permute_ps" | "_mm512_permute_ps" | "_mm512_mask_permute_ps" => "vpermilps",
        "_mm256_shuffle_pd" | "_mm512_shuffle_pd" | "_mm512_mask_shuffle_pd" => "vshufpd",
        "_mm256_shuffle_ps" | "_mm512_shuffle_ps" | "_mm512_mask_shuffle_ps" => "vshufps",
        "_mm256_permutexvar_epi32" | "_mm512_permutexvar_epi32" => "vpermd",
        "_mm512_permutex2var_epi32" | "_mm512_mask_permutex2var_epi32" => "vpermi2d",
        "_mm512_permutex2var_epi64" | "_mm512_mask_permutex2var_epi64" => "vpermi2q",
        "_mm256_unpacklo_epi32" | "_mm512_unpacklo_epi32" | "_mm512_mask_unpacklo_epi32" => {
            "vpunpckldq"
        }
        "_mm256_unpackhi_epi32" | "_mm512_unpackhi_epi32" | "_mm512_mask_unpackhi_epi32" => {
            "vpunpckhdq"
        }
        "_mm256_unpacklo_epi64" | "_mm512_unpacklo_epi64" | "_mm512_mask_unpacklo_epi64" => {
            "vpunpcklqdq"
        }
        "_mm256_unpackhi_epi64" | "_mm512_unpackhi_epi64" | "_mm512_mask_unpackhi_epi64" => {
            "vpunpckhqdq"
        }
        "_mm256_permute2x128_si256" => "vperm2i128",
        "_mm512_shuffle_i32x4" | "_mm512_mask_shuffle_i32x4" => "vshufi32x4",
        "_mm256_alignr_epi8"
        | "_mm256_alignr_epi32"
        | "_mm512_alignr_epi32"
        | "_mm512_mask_alignr_epi32" => "valignd",
        "_mm256_alignr_epi64" | "_mm512_alignr_epi64" | "_mm512_mask_alignr_epi64" => "valignq",
        "_mm256_blend_pd" => "vblendpd",
        "_mm256_blend_ps" => "vblendps",
        _ => "unsupported",
    }
}

// ---------------------------------------------------------------------------
// Constant pooling / materialization used by lowering before rendering.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ConstantPool {
    constants: Vec<ConstantData>,
    labels_by_key: BTreeMap<ConstantKey, String>,
}

impl ConstantPool {
    fn label_for_key(&mut self, key: &ConstantKey) -> String {
        if let Some(label) = self.labels_by_key.get(key) {
            return label.clone();
        }

        let label = format!("LC{}", self.constants.len());
        self.constants.push(ConstantData {
            label: label.clone(),
            key: key.clone(),
        });
        self.labels_by_key.insert(key.clone(), label.clone());
        label
    }

    fn into_constants(self) -> Vec<ConstantData> {
        self.constants
    }
}

#[derive(Default)]
struct MaterializedConstants {
    vector_registers: BTreeMap<ConstantKey, Register>,
    kmask_registers: BTreeMap<ConstantKey, u8>,
    next_kmask: u8,
}

impl MaterializedConstants {
    fn materialize_vector(
        &mut self,
        instructions: &mut Vec<ModeledInstruction>,
        constants: &mut ConstantPool,
        regs: &mut LoweringRegAlloc,
        bits: u32,
        hex: &str,
    ) -> Register {
        let key = ConstantKey::Vector {
            bits,
            hex: hex.to_owned(),
        };
        if let Some(register) = self.vector_registers.get(&key) {
            return register.clone();
        }

        let label = constants.label_for_key(&key);
        let register = regs.allocate_persistent_temp();
        instructions.push(ModeledInstruction {
            mnemonic: vector_constant_load_mnemonic(bits).to_owned(),
            operands: vec![
                Operand::Register(register.clone()),
                Operand::ConstantRef(label),
            ],
            comment: Some("materialize vector constant".to_owned()),
        });
        self.vector_registers.insert(key, register.clone());
        register
    }

    fn materialize_kmask(
        &mut self,
        instructions: &mut Vec<ModeledInstruction>,
        _constants: &mut ConstantPool,
        bits: u32,
        value: u64,
    ) -> u8 {
        let key = ConstantKey::KMask { bits, value };
        if let Some(register) = self.kmask_registers.get(&key) {
            return *register;
        }

        let register = self.allocate_kmask();
        let gpr = if bits <= 32 { "eax" } else { "rax" };
        instructions.push(ModeledInstruction {
            mnemonic: "mov".to_owned(),
            operands: vec![Operand::Gpr(gpr), Operand::Immediate(value)],
            comment: Some("materialize k-mask constant".to_owned()),
        });
        instructions.push(ModeledInstruction {
            mnemonic: kmask_load_mnemonic(bits).to_owned(),
            operands: vec![Operand::KMask(register), Operand::Gpr(gpr)],
            comment: Some("materialize k-mask constant".to_owned()),
        });
        self.kmask_registers.insert(key, register);
        register
    }

    fn materialize_control_vector(
        &mut self,
        instructions: &mut Vec<ModeledInstruction>,
        constants: &mut ConstantPool,
        regs: &mut LoweringRegAlloc,
        bits: u32,
        element_bits: u32,
        value: u64,
    ) -> Register {
        let key = ConstantKey::ControlVector {
            bits,
            element_bits,
            value,
        };
        if let Some(register) = self.vector_registers.get(&key) {
            return register.clone();
        }

        let label = constants.label_for_key(&key);
        let register = regs.allocate_persistent_temp();
        instructions.push(ModeledInstruction {
            mnemonic: vector_constant_load_mnemonic(element_bits).to_owned(),
            operands: vec![
                Operand::Register(register.clone()),
                Operand::ConstantRef(label),
            ],
            comment: Some(format_control_vector(value, bits, element_bits)),
        });
        self.vector_registers.insert(key, register.clone());
        register
    }

    fn allocate_kmask(&mut self) -> u8 {
        let register = self.next_kmask + 1;
        self.next_kmask += 1;
        register
    }
}

fn vector_move_mnemonic(arch: ArchArg) -> &'static str {
    match arch {
        ArchArg::Avx2 => "vmovdqa32",
        ArchArg::Avx512 => "vmovdqa64",
    }
}

fn vector_constant_load_mnemonic(bits: u32) -> &'static str {
    if bits.is_multiple_of(64) {
        "vmovdqa64"
    } else {
        "vmovdqa32"
    }
}

fn kmask_load_mnemonic(bits: u32) -> &'static str {
    if bits <= 16 { "kmovw" } else { "kmovq" }
}

// ---------------------------------------------------------------------------
// Internal register allocator used by lowering before rendering.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum LoweringSide {
    Top,
    Bottom,
}

struct LoweringRegAlloc {
    arch: ArchArg,
    num_vecs: usize,
    pair_index: usize,
    next_temp: usize,
    temp_floor: usize,
}

impl LoweringRegAlloc {
    fn new(arch: ArchArg, num_vecs: usize) -> Self {
        let temp_base = num_vecs * 2;
        Self {
            arch,
            num_vecs,
            pair_index: 0,
            next_temp: temp_base,
            temp_floor: temp_base,
        }
    }

    fn reg(&self, index: usize) -> Register {
        match self.arch {
            ArchArg::Avx2 => Register::Ymm(index as u8),
            ArchArg::Avx512 => Register::Zmm(index as u8),
        }
    }

    fn src_top(&self) -> Register {
        if self.pair_index == 0 {
            self.reg(0)
        } else {
            self.reg(self.num_vecs)
        }
    }

    fn src_bottom(&self) -> Register {
        if self.pair_index == 0 {
            self.reg(1)
        } else {
            self.reg(self.num_vecs + 1)
        }
    }

    fn dst_top(&self) -> Register {
        if self.pair_index == 0 {
            self.reg(self.num_vecs)
        } else {
            self.reg(0)
        }
    }

    fn dst_bottom(&self) -> Register {
        if self.pair_index == 0 {
            self.reg(self.num_vecs + 1)
        } else {
            self.reg(1)
        }
    }

    fn source_for_side(&self, side: LoweringSide) -> Register {
        match side {
            LoweringSide::Top => self.src_top(),
            LoweringSide::Bottom => self.src_bottom(),
        }
    }

    fn destination_for_side(&self, side: LoweringSide) -> Register {
        match side {
            LoweringSide::Top => self.dst_top(),
            LoweringSide::Bottom => self.dst_bottom(),
        }
    }

    fn allocate_temp(&mut self) -> Register {
        let r = self.reg(self.next_temp);
        self.next_temp += 1;
        r
    }

    fn allocate_persistent_temp(&mut self) -> Register {
        let r = self.allocate_temp();
        self.temp_floor = self.next_temp;
        r
    }

    fn reset_temps(&mut self) {
        self.next_temp = self.temp_floor;
    }

    fn swap_pairs(&mut self) {
        self.pair_index = 1 - self.pair_index;
    }
}
