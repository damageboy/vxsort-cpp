use std::collections::BTreeMap;

use bitonic_codegen::instruction_stream::{
    ConstantData, ConstantKey, InstructionBlock, InstructionStream, LoweringOptions,
    ModeledInstruction, Operand, Register, lower_solution_paths,
};
use bitonic_codegen::json_exporter::SolutionJsonMetadata;
use bitonic_codegen::transition_table::TransitionTable;
use bitonic_codegen::{ArchArg, DTypeArg};
use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget, VectorState};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn state(top: &[u8], bottom: &[u8]) -> VectorState {
    VectorState::new(top.to_vec(), bottom.to_vec())
}

fn path_for(
    table: &TransitionTable,
    stage: usize,
    input: &VectorState,
    output: &VectorState,
) -> bitonic_codegen::transition_table::CompletePath {
    table
        .complete_path_from_zero_based_steps(&[(stage, input.as_tuple(), output.as_tuple())])
        .expect("test path should resolve")
}

fn inst(name: &'static str, args: &[(&'static str, InstructionArg)]) -> InstructionSpec {
    InstructionSpec::new(name, args.iter().cloned().collect::<BTreeMap<_, _>>())
}

fn avx2_i64_metadata() -> SolutionJsonMetadata {
    SolutionJsonMetadata {
        natural_order: false,
        arch: ArchArg::Avx2,
        dtype: DTypeArg::I64,
        num_vecs: 2,
    }
}

fn avx2_i32_metadata() -> SolutionJsonMetadata {
    SolutionJsonMetadata {
        natural_order: false,
        arch: ArchArg::Avx2,
        dtype: DTypeArg::I32,
        num_vecs: 2,
    }
}

fn avx512_i64_metadata() -> SolutionJsonMetadata {
    SolutionJsonMetadata {
        natural_order: false,
        arch: ArchArg::Avx512,
        dtype: DTypeArg::I64,
        num_vecs: 2,
    }
}

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

#[test]
fn construct_stream_with_single_block_and_instruction() {
    let stream = InstructionStream {
        constants: vec![],
        blocks: vec![InstructionBlock {
            label: Some("vxsort_solution_0".to_owned()),
            instructions: vec![ModeledInstruction {
                mnemonic: "vpermq".to_owned(),
                operands: vec![Operand::Register(Register::Ymm(0)), Operand::Immediate(78)],
                comment: Some("stage 0".to_owned()),
            }],
        }],
    };

    assert_eq!(stream.blocks.len(), 1);
    assert_eq!(stream.constants.len(), 0);

    let block = &stream.blocks[0];
    assert_eq!(block.label.as_deref(), Some("vxsort_solution_0"));
    assert_eq!(block.instructions.len(), 1);

    let instr = &block.instructions[0];
    assert_eq!(instr.mnemonic, "vpermq");
    assert_eq!(instr.comment.as_deref(), Some("stage 0"));
    assert_eq!(instr.operands.len(), 2);
    assert_eq!(instr.operands[0], Operand::Register(Register::Ymm(0)));
    assert_eq!(instr.operands[1], Operand::Immediate(78));
}

#[test]
fn stream_equality_holds_for_identical_instances() {
    let make = || InstructionStream {
        constants: vec![],
        blocks: vec![InstructionBlock {
            label: Some("vxsort_solution_0".to_owned()),
            instructions: vec![ModeledInstruction {
                mnemonic: "vpermq".to_owned(),
                operands: vec![Operand::Register(Register::Ymm(0)), Operand::Immediate(78)],
                comment: Some("stage 0".to_owned()),
            }],
        }],
    };
    assert_eq!(make(), make());
}

#[test]
fn stream_clone_is_independent() {
    let original = InstructionStream {
        constants: vec![],
        blocks: vec![InstructionBlock {
            label: Some("lbl".to_owned()),
            instructions: vec![],
        }],
    };
    let mut cloned = original.clone();
    cloned.blocks[0].label = Some("other".to_owned());
    assert_eq!(original.blocks[0].label.as_deref(), Some("lbl"));
}

// ---------------------------------------------------------------------------
// Register helper tests
// ---------------------------------------------------------------------------

#[test]
fn register_ymm_name() {
    assert_eq!(Register::Ymm(0).name(), "ymm0");
    assert_eq!(Register::Ymm(15).name(), "ymm15");
}

#[test]
fn register_zmm_name() {
    assert_eq!(Register::Zmm(3).name(), "zmm3");
    assert_eq!(Register::Zmm(31).name(), "zmm31");
}

#[test]
fn register_display_matches_name() {
    for r in [Register::Ymm(5), Register::Zmm(1)] {
        assert_eq!(format!("{r}"), r.name());
    }
}

// ---------------------------------------------------------------------------
// Operand variant tests
// ---------------------------------------------------------------------------

#[test]
fn operand_variants_constructible() {
    let _ = Operand::Register(Register::Ymm(0));
    let _ = Operand::Immediate(42);
    let _ = Operand::ConstantRef("my_const".to_owned());
    let _ = Operand::KMask(2);
    let _ = Operand::Unsupported("unknown_intrinsic".to_owned());
}

#[test]
fn operand_eq() {
    assert_eq!(Operand::Immediate(78), Operand::Immediate(78));
    assert_ne!(Operand::Immediate(1), Operand::Immediate(2));
    assert_ne!(
        Operand::Register(Register::Ymm(0)),
        Operand::Register(Register::Zmm(0))
    );
}

// ---------------------------------------------------------------------------
// ConstantData / ConstantKey tests
// ---------------------------------------------------------------------------

#[test]
fn constant_key_equality() {
    assert_eq!(
        ConstantKey::Vector {
            bits: 256,
            hex: "0x1234".to_owned(),
        },
        ConstantKey::Vector {
            bits: 256,
            hex: "0x1234".to_owned(),
        }
    );
    assert_ne!(
        ConstantKey::Vector {
            bits: 256,
            hex: "0x1234".to_owned(),
        },
        ConstantKey::KMask {
            bits: 16,
            value: 0x1234,
        }
    );
}

#[test]
fn constant_data_round_trip() {
    let key = ConstantKey::Vector {
        bits: 256,
        hex: "0x1234".to_owned(),
    };
    let data = ConstantData {
        label: "LC0".to_owned(),
        key: key.clone(),
    };
    assert_eq!(data.label, "LC0");
    assert_eq!(data.key, key);
}

// ---------------------------------------------------------------------------
// Constant materialization / GVN tests
// ---------------------------------------------------------------------------

#[test]
fn identical_bitvec_constants_share_one_pool_entry_and_materialized_register() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let repeated_constant = InstructionArg::BitVec {
        bits: 256,
        hex: "0x01020304".to_owned(),
    };
    let gadget = PermutationGadget::new(
        vec![
            inst(
                "_mm512_permutex2var_epi64",
                &[
                    ("a", InstructionArg::Input("top".to_owned())),
                    ("b", repeated_constant.clone()),
                ],
            ),
            inst(
                "_mm512_permutex2var_epi64",
                &[
                    ("a", InstructionArg::Input("prev".to_owned())),
                    ("b", repeated_constant),
                ],
            ),
        ],
        vec![],
    );

    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, 0, &input, &output);

    let stream = lower_solution_paths(
        &avx512_i64_metadata(),
        &table,
        &[path],
        LoweringOptions::default(),
    );

    assert_eq!(stream.constants.len(), 1);
    assert_eq!(stream.constants[0].label, "LC0");
    assert_eq!(
        stream.constants[0].key,
        ConstantKey::Vector {
            bits: 256,
            hex: "0x01020304".to_owned(),
        }
    );

    let block = &stream.blocks[0];
    let materializations: Vec<_> = block
        .instructions
        .iter()
        .filter(|instruction| instruction.mnemonic == "vmovdqa64")
        .collect();
    assert_eq!(materializations.len(), 1);
    assert_eq!(
        materializations[0].operands[1],
        Operand::ConstantRef("LC0".to_owned())
    );

    let materialized_register = match &materializations[0].operands[0] {
        Operand::Register(register) => register.clone(),
        other => panic!("expected materialization register, got {other:?}"),
    };
    let uses = block
        .instructions
        .iter()
        .filter(|instruction| instruction.mnemonic == "vpermi2q")
        .filter(|instruction| {
            instruction
                .operands
                .contains(&Operand::Register(materialized_register.clone()))
        })
        .count();
    assert_eq!(uses, 2);
}

#[test]
fn lowering_options_can_omit_human_comments_for_scoring() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let gadget = PermutationGadget::new(
        vec![inst(
            "_mm256_permute4x64_epi64",
            &[
                ("a", InstructionArg::Input("top".to_owned())),
                ("imm8", InstructionArg::U64(0x1b)),
            ],
        )],
        vec![],
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, 0, &input, &output);

    let stream = lower_solution_paths(
        &avx2_i64_metadata(),
        &table,
        &[path],
        LoweringOptions {
            include_comments: false,
        },
    );

    let block = &stream.blocks[0];
    assert!(
        block
            .instructions
            .iter()
            .all(|instruction| !instruction.mnemonic.is_empty() && instruction.comment.is_none()),
        "scoring lowering should not build comment-only state tables: {:?}",
        block.instructions
    );
}

#[test]
fn identical_kmask_constants_share_one_pool_entry_and_materialized_register() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let gadget = PermutationGadget::new(
        vec![
            inst(
                "_mm512_mask_permute_pd",
                &[
                    ("a", InstructionArg::Input("top".to_owned())),
                    ("k", InstructionArg::U64(0x00ff)),
                    ("imm8", InstructionArg::U64(0x55)),
                ],
            ),
            inst(
                "_mm512_mask_permute_pd",
                &[
                    ("a", InstructionArg::Input("prev".to_owned())),
                    ("k", InstructionArg::U64(0x00ff)),
                    ("imm8", InstructionArg::U64(0xaa)),
                ],
            ),
        ],
        vec![],
    );

    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, 0, &input, &output);

    let stream = lower_solution_paths(
        &avx512_i64_metadata(),
        &table,
        &[path],
        LoweringOptions::default(),
    );

    assert!(stream.constants.is_empty());

    let block = &stream.blocks[0];
    let mov_count = block
        .instructions
        .iter()
        .filter(|instruction| instruction.mnemonic == "mov")
        .count();
    assert_eq!(mov_count, 1);

    let kmovw_count = block
        .instructions
        .iter()
        .filter(|instruction| instruction.mnemonic == "kmovw")
        .count();
    assert_eq!(kmovw_count, 1);

    let k1_uses = block
        .instructions
        .iter()
        .filter(|instruction| instruction.mnemonic == "vpermilpd")
        .filter(|instruction| {
            instruction
                .operands
                .iter()
                .any(|operand| matches!(operand, Operand::MaskedRegister(_, 1)))
        })
        .count();
    assert_eq!(k1_uses, 2);
}

// ---------------------------------------------------------------------------
// InstructionBlock: unlabeled block
// ---------------------------------------------------------------------------

#[test]
fn block_without_label() {
    let block = InstructionBlock {
        label: None,
        instructions: vec![],
    };
    assert!(block.label.is_none());
}

// ---------------------------------------------------------------------------
// Lowering tests (Task 2)
// ---------------------------------------------------------------------------

/// Build a one-stage AVX2 i64 path using `_mm256_permute4x64_epi64`,
/// lower it, and verify the resulting modeled instruction sequence.
#[test]
fn lower_one_stage_avx2_i64_path() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let gadget = PermutationGadget::new(
        vec![inst(
            "_mm256_permute4x64_epi64",
            &[
                ("a", InstructionArg::Input("top".to_owned())),
                ("imm8", InstructionArg::U64(0x4e)),
            ],
        )],
        vec![],
    );

    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, 0, &input, &output);

    let stream = lower_solution_paths(
        &avx2_i64_metadata(),
        &table,
        &[path],
        LoweringOptions::default(),
    );

    // One block, no constant pool entries
    assert_eq!(stream.blocks.len(), 1);
    assert!(stream.constants.is_empty());

    let block = &stream.blocks[0];
    assert_eq!(block.label.as_deref(), Some("vxsort_solution_0"));

    let mnemonics: Vec<&str> = block
        .instructions
        .iter()
        .map(|i| i.mnemonic.as_str())
        .collect();

    // Permutation gadget produces vpermq
    assert!(
        mnemonics.contains(&"vpermq"),
        "expected vpermq, got {mnemonics:?}"
    );

    // AVX2 i64 compare-swap: vpcmpgtq + two vblendvpd
    assert!(
        mnemonics.contains(&"vpcmpgtq"),
        "expected vpcmpgtq, got {mnemonics:?}"
    );
    let vblendvpd_count = mnemonics.iter().filter(|&&m| m == "vblendvpd").count();
    assert_eq!(
        vblendvpd_count, 2,
        "expected 2 vblendvpd, got {vblendvpd_count} in {mnemonics:?}"
    );

    // Final instruction must be ret
    assert_eq!(
        mnemonics.last(),
        Some(&"ret"),
        "expected ret as final instruction, got {mnemonics:?}"
    );
}

/// The vpermq operands match: dest=ymm0, src=ymm0, imm=78 (0x4e).
#[test]
fn lower_avx2_i64_vpermq_operands() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let gadget = PermutationGadget::new(
        vec![inst(
            "_mm256_permute4x64_epi64",
            &[
                ("a", InstructionArg::Input("top".to_owned())),
                ("imm8", InstructionArg::U64(0x4e)),
            ],
        )],
        vec![],
    );

    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, 0, &input, &output);

    let stream = lower_solution_paths(
        &avx2_i64_metadata(),
        &table,
        &[path],
        LoweringOptions::default(),
    );

    let block = &stream.blocks[0];
    let vpermq = block
        .instructions
        .iter()
        .find(|i| i.mnemonic == "vpermq")
        .expect("vpermq not found");

    // Intel syntax: dest, src, imm  →  ymm0, ymm0, 78
    assert_eq!(
        vpermq.operands,
        vec![
            Operand::Register(Register::Ymm(0)),
            Operand::Register(Register::Ymm(0)),
            Operand::Immediate(78),
        ],
        "unexpected vpermq operands: {:?}",
        vpermq.operands
    );
}

#[test]
fn lower_avx2_i32_blacher_shuffle_epi32_mnemonic() {
    let input = state(&[1, 3, 5, 7, 9, 11, 13, 15], &[2, 4, 6, 8, 10, 12, 14, 16]);
    let output = input.clone();
    let gadget = PermutationGadget::new(
        vec![inst(
            "_mm256_shuffle_epi32",
            &[
                ("a", InstructionArg::Input("top".to_owned())),
                ("imm8", InstructionArg::U64(0xb1)),
            ],
        )],
        Vec::new(),
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, 0, &input, &output);

    let stream = lower_solution_paths(
        &avx2_i32_metadata(),
        &table,
        &[path],
        LoweringOptions {
            include_comments: false,
        },
    );

    let mnemonics: Vec<&str> = stream.blocks[0]
        .instructions
        .iter()
        .map(|instruction| instruction.mnemonic.as_str())
        .collect();
    assert!(mnemonics.contains(&"vpshufd"), "{mnemonics:?}");
    assert!(!mnemonics.contains(&"unsupported"), "{mnemonics:?}");
}

/// Extra operands that lowering does not yet support stay explicit in the IR.
#[test]
fn lower_unconsumed_operands_as_unsupported_markers() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let gadget = PermutationGadget::new(
        vec![inst(
            "_mm512_mask_permutex2var_epi64",
            &[
                ("a", InstructionArg::Input("top".to_owned())),
                ("b", InstructionArg::Input("bottom".to_owned())),
                ("k", InstructionArg::U64(0xff)),
                (
                    "op_idx",
                    InstructionArg::BitVec {
                        bits: 512,
                        hex: "0x1234".to_owned(),
                    },
                ),
            ],
        )],
        vec![],
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, 0, &input, &output);

    let stream = lower_solution_paths(
        &avx512_i64_metadata(),
        &table,
        &[path],
        LoweringOptions::default(),
    );

    let instruction = stream.blocks[0]
        .instructions
        .iter()
        .find(|instruction| instruction.mnemonic == "vpermi2q")
        .expect("masked permutex2var instruction should be lowered");
    assert!(
        instruction
            .operands
            .iter()
            .any(|operand| matches!(operand, Operand::MaskedRegister(_, 1)))
    );
    assert!(instruction.operands.iter().any(|operand| {
        matches!(operand, Operand::Unsupported(text) if text.contains("unconsumed operand op_idx"))
    }));
}

#[test]
fn lower_src_alias_does_not_hide_unconsumed_a_operand() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let gadget = PermutationGadget::new(
        vec![inst(
            "_mm256_permute4x64_epi64",
            &[
                ("src", InstructionArg::Input("top".to_owned())),
                ("a", InstructionArg::Input("bottom".to_owned())),
                ("imm8", InstructionArg::U64(0x4e)),
            ],
        )],
        vec![],
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, 0, &input, &output);

    let stream = lower_solution_paths(
        &avx2_i64_metadata(),
        &table,
        &[path],
        LoweringOptions::default(),
    );

    let instruction = stream.blocks[0]
        .instructions
        .iter()
        .find(|instruction| instruction.mnemonic == "vpermq")
        .expect("vpermq should be lowered");
    assert!(instruction.operands.iter().any(|operand| {
        matches!(operand, Operand::Unsupported(text) if text.contains("unconsumed operand src"))
    }));
}

/// An empty path list produces an empty stream.
#[test]
fn lower_empty_paths_produces_empty_stream() {
    let table = TransitionTable::new(0);
    let stream = lower_solution_paths(
        &avx2_i64_metadata(),
        &table,
        &[],
        LoweringOptions::default(),
    );
    assert!(stream.blocks.is_empty());
    assert!(stream.constants.is_empty());
}

/// A gadget with no matching transition emits an `Operand::Unsupported` marker
/// instead of silently dropping instructions.
#[test]
fn id_based_path_resolution_rejects_missing_transition() {
    // Build a table for stage 0 but register a *different* input/output pair
    // than what the path references, so the lookup misses.
    let registered_input = state(&[0, 1, 2, 3], &[4, 5, 6, 7]);
    let registered_output = state(&[4, 5, 6, 7], &[0, 1, 2, 3]);
    let gadget = PermutationGadget::new(
        vec![inst(
            "_mm256_permute4x64_epi64",
            &[
                ("a", InstructionArg::Input("top".to_owned())),
                ("imm8", InstructionArg::U64(0x1b)),
            ],
        )],
        vec![],
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &registered_input, &registered_output, gadget);

    // Path uses a *different* transition that isn't in the table
    let path_input = state(&[9, 8, 7, 6], &[5, 4, 3, 2]);
    let path_output = state(&[5, 4, 3, 2], &[9, 8, 7, 6]);
    assert!(
        table
            .complete_path_from_zero_based_steps(&[(
                0,
                path_input.as_tuple(),
                path_output.as_tuple()
            )])
            .is_none()
    );
}
