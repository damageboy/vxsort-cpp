use std::collections::BTreeMap;

use bitonic_codegen::asm_exporter::generate_solution_asm_for_paths;
use bitonic_codegen::json_exporter::SolutionJsonMetadata;
use bitonic_codegen::transition_table::TransitionTable;
use bitonic_codegen::{ArchArg, DTypeArg};
use gadget_synth::{InstructionArg, InstructionSpec, PermutationGadget, VectorState};

fn state(top: &[u8], bottom: &[u8]) -> VectorState {
    VectorState::new(top.to_vec(), bottom.to_vec())
}

fn path_for(
    table: &TransitionTable,
    steps: &[(usize, &VectorState, &VectorState)],
) -> bitonic_codegen::transition_table::CompletePath {
    let steps = steps
        .iter()
        .map(|(stage, input, output)| (*stage, input.as_tuple(), output.as_tuple()))
        .collect::<Vec<_>>();
    table
        .complete_path_from_zero_based_steps(&steps)
        .expect("test path should resolve")
}

fn inst(name: &'static str, args: &[(&'static str, InstructionArg)]) -> InstructionSpec {
    InstructionSpec::new(name, args.iter().cloned().collect::<BTreeMap<_, _>>())
}

fn metadata() -> SolutionJsonMetadata {
    metadata_for(ArchArg::Avx2, DTypeArg::I64)
}

fn metadata_for(arch: ArchArg, dtype: DTypeArg) -> SolutionJsonMetadata {
    SolutionJsonMetadata {
        natural_order: false,
        arch,
        dtype,
        num_vecs: 2,
    }
}

fn compare_swap_asm_for(arch: ArchArg, dtype: DTypeArg) -> String {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let gadget = PermutationGadget::new(Vec::new(), Vec::new());
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, &[(0, &input, &output)]);

    generate_solution_asm_for_paths(&metadata_for(arch, dtype), &table, &[path])
}

fn assert_mnemonic_count(asm: &str, mnemonic: &str, expected: usize) {
    assert_eq!(
        asm.matches(mnemonic).count(),
        expected,
        "expected {expected} occurrence(s) of {mnemonic} in:\n{asm}"
    );
}

#[test]
fn emits_nasm_style_solution_with_gadget_and_compare_swap() {
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
        Vec::new(),
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, &[(0, &input, &output)]);

    let asm = generate_solution_asm_for_paths(&metadata(), &table, &[path]);

    assert!(asm.contains("section .text"));
    assert!(asm.contains("global vxsort_solution_0"));
    assert!(asm.contains("vxsort_solution_0:"));
    assert!(asm.contains("; stage 0"));
    assert!(asm.contains("vpermq"));
    assert!(asm.contains("ymm0, ymm0, 0x4e"));
    assert!(asm.contains("_MM_SHUFFLE(1, 0, 3, 2)"));
    assert!(asm.contains("vpcmpgtq"));
    assert!(asm.contains("vblendvpd"));
    assert!(asm.contains("ret"));
}

#[test]
fn imm8_comments_follow_python_intrinsic_metadata() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let gadget = PermutationGadget::new(
        vec![
            inst(
                "_mm256_shuffle_pd",
                &[
                    ("a", InstructionArg::Input("top".to_owned())),
                    ("b", InstructionArg::Input("bottom".to_owned())),
                    ("imm8", InstructionArg::U64(0x0a)),
                ],
            ),
            inst(
                "_mm256_blend_pd",
                &[
                    ("a", InstructionArg::Input("prev".to_owned())),
                    ("b", InstructionArg::Input("bottom".to_owned())),
                    ("imm8", InstructionArg::U64(0x05)),
                ],
            ),
        ],
        Vec::new(),
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, &[(0, &input, &output)]);

    let asm = generate_solution_asm_for_paths(&metadata(), &table, &[path]);

    assert!(asm.contains("_MM_SHUFFLE2(2, 2)"));
    assert!(asm.contains("0b00000101"));
    assert!(!asm.contains("imm8=0b00001010"));
}

#[test]
fn emits_comfy_stage_state_register_and_compare_swap_comments() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let gadget = PermutationGadget::new(Vec::new(), Vec::new());
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, &[(0, &input, &output)]);

    let asm = generate_solution_asm_for_paths(&metadata(), &table, &[path]);

    assert!(asm.contains("; Initial Input State:"));
    assert!(asm.contains("; ╭────────┬───┬───┬───┬───╮"));
    assert!(asm.contains("; │        ┆ 0 ┆ 1 ┆ 2 ┆ 3 │"));
    assert!(asm.contains("; ╞════════╪═══╪═══╪═══╪═══╡"));
    assert!(asm.contains("; │ Top    ┆ 1 ┆ 3 ┆ 5 ┆ 7 │"));
    assert!(asm.contains("; │ Bottom ┆ 2 ┆ 4 ┆ 6 ┆ 8 │"));
    assert!(asm.contains("; ╰────────┴───┴───┴───┴───╯"));
    assert!(asm.contains("; Stage 0"));
    assert!(!asm.contains("; Registers:"));
    assert!(!asm.contains("; Compare-swap:"));
    assert!(asm.contains("; Output State:"));
    assert!(asm.contains("; │ Top    ┆ 1 ┆ 2 ┆ 5 ┆ 6 │"));
    assert!(asm.contains("; │ Bottom ┆ 3 ┆ 4 ┆ 7 ┆ 8 │"));
}

#[test]
fn emits_current_top_bottom_register_comment_at_each_stage() {
    let input0 = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output0 = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let output1 = state(&[1, 2, 3, 4], &[5, 6, 7, 8]);
    let gadget = PermutationGadget::new(Vec::new(), Vec::new());
    let mut table = TransitionTable::new(2);
    table.add_transition(0, &input0, &output0, gadget.clone());
    table.add_transition(1, &output0, &output1, gadget);
    let path = path_for(&table, &[(0, &input0, &output0), (1, &output0, &output1)]);

    let asm = generate_solution_asm_for_paths(&metadata(), &table, &[path]);

    assert!(asm.contains("; Stage 0\n    ; Current registers: ymm0 = top, ymm1 = bottom"));
    assert!(asm.contains("; Stage 1\n    ; Current registers: ymm2 = top, ymm3 = bottom"));
}

#[test]
fn avx2_i64_compare_swap_uses_cmpgtq_and_two_blendvpd() {
    let asm = compare_swap_asm_for(ArchArg::Avx2, DTypeArg::I64);

    assert_mnemonic_count(&asm, "vpcmpgtq", 1);
    assert_mnemonic_count(&asm, "vblendvpd", 2);
    assert_mnemonic_count(&asm, "vpminsd", 0);
    assert_mnemonic_count(&asm, "vpmaxsd", 0);
    assert_mnemonic_count(&asm, "vpminsq", 0);
    assert_mnemonic_count(&asm, "vpmaxsq", 0);
}

#[test]
fn avx2_i32_compare_swap_uses_minmax_sd() {
    let asm = compare_swap_asm_for(ArchArg::Avx2, DTypeArg::I32);

    assert_mnemonic_count(&asm, "vpminsd", 1);
    assert_mnemonic_count(&asm, "vpmaxsd", 1);
    assert_mnemonic_count(&asm, "vpcmpgtq", 0);
    assert_mnemonic_count(&asm, "vblendvpd", 0);
    assert_mnemonic_count(&asm, "vpminsq", 0);
    assert_mnemonic_count(&asm, "vpmaxsq", 0);
}

#[test]
fn avx2_i16_compare_swap_uses_minmax_sw() {
    let asm = compare_swap_asm_for(ArchArg::Avx2, DTypeArg::I16);

    assert_mnemonic_count(&asm, "vpminsw", 1);
    assert_mnemonic_count(&asm, "vpmaxsw", 1);
    assert_mnemonic_count(&asm, "vpcmpgtq", 0);
    assert_mnemonic_count(&asm, "vblendvpd", 0);
}

#[test]
fn avx2_u64_compare_swap_matches_python_alias_emulation() {
    let asm = compare_swap_asm_for(ArchArg::Avx2, DTypeArg::U64);

    assert_mnemonic_count(&asm, "vpcmpgtq", 1);
    assert_mnemonic_count(&asm, "vblendvpd", 2);
    assert_mnemonic_count(&asm, "vpminsq", 0);
    assert_mnemonic_count(&asm, "vpmaxsq", 0);
}

#[test]
fn avx512_i64_compare_swap_uses_minmax_sq() {
    let asm = compare_swap_asm_for(ArchArg::Avx512, DTypeArg::I64);

    assert_mnemonic_count(&asm, "vpminsq", 1);
    assert_mnemonic_count(&asm, "vpmaxsq", 1);
    assert_mnemonic_count(&asm, "vpcmpgtq", 0);
    assert_mnemonic_count(&asm, "vblendvpd", 0);
    assert_mnemonic_count(&asm, "vpminsd", 0);
    assert_mnemonic_count(&asm, "vpmaxsd", 0);
}

#[test]
fn avx512_u32_compare_swap_matches_i32_minmax() {
    let asm = compare_swap_asm_for(ArchArg::Avx512, DTypeArg::U32);

    assert_mnemonic_count(&asm, "vpminsd", 1);
    assert_mnemonic_count(&asm, "vpmaxsd", 1);
    assert_mnemonic_count(&asm, "vpcmpgtq", 0);
    assert_mnemonic_count(&asm, "vblendvpd", 0);
}

#[test]
fn unsupported_gadget_instruction_forms_remain_explicit_in_asm() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let gadget = PermutationGadget::new(
        vec![inst(
            "_mm256_not_supported",
            &[("imm8", InstructionArg::U64(1))],
        )],
        Vec::new(),
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, &[(0, &input, &output)]);

    let asm = generate_solution_asm_for_paths(&metadata(), &table, &[path]);

    assert!(
        asm.contains("unsupported"),
        "ASM omitted unsupported marker:\n{asm}"
    );
    assert!(
        asm.contains("_mm256_not_supported"),
        "ASM omitted unsupported intrinsic detail:\n{asm}"
    );
    assert!(asm.contains("ret"));
}

#[test]
fn emits_rodata_for_stream_constants_without_renderer_pooling() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let constant = InstructionArg::BitVec {
        bits: 256,
        hex: "0x01020304".to_owned(),
    };
    let gadget = PermutationGadget::new(
        vec![
            inst(
                "_mm512_permutex2var_epi64",
                &[
                    ("a", InstructionArg::Input("top".to_owned())),
                    ("b", constant.clone()),
                ],
            ),
            inst(
                "_mm512_permutex2var_epi64",
                &[
                    ("a", InstructionArg::Input("prev".to_owned())),
                    ("b", constant),
                ],
            ),
        ],
        Vec::new(),
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, &[(0, &input, &output)]);

    let asm = generate_solution_asm_for_paths(&metadata(), &table, &[path]);

    assert!(asm.contains("section .rodata"));
    assert!(asm.contains("LC0:"));
    assert!(asm.contains("db                 0x04, 0x03, 0x02, 0x01"));
    assert!(asm.contains("vmovdqa64"));
    assert!(asm.contains("[rel LC0]"));
    assert_eq!(asm.matches("LC0:").count(), 1);
    assert_eq!(asm.matches("[rel LC0]").count(), 1);
}

#[test]
fn emits_control_vectors_as_rodata_with_lane_comments() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let gadget = PermutationGadget::new(
        vec![inst(
            "_mm256_permutexvar_epi64",
            &[
                ("a", InstructionArg::Input("top".to_owned())),
                ("b", InstructionArg::U64(0x03020100)),
            ],
        )],
        Vec::new(),
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, &[(0, &input, &output)]);

    let asm = generate_solution_asm_for_paths(&metadata(), &table, &[path]);

    assert!(asm.contains("section .rodata"));
    assert!(asm.contains("dq                 0, 1, 2, 3"));
    assert!(asm.contains("vmovdqa64"));
    assert!(asm.contains("[0, 1, 2, 3]"));
    assert!(asm.contains("vpermq"));
}

#[test]
fn emits_mask_loads_and_writemask_operands_for_avx512() {
    let input = state(&[1, 3, 5, 7, 9, 11, 13, 15], &[2, 4, 6, 8, 10, 12, 14, 16]);
    let output = state(&[1, 2, 5, 6, 9, 10, 13, 14], &[3, 4, 7, 8, 11, 12, 15, 16]);
    let gadget = PermutationGadget::new(
        vec![inst(
            "_mm512_mask_shuffle_i32x4",
            &[
                ("a", InstructionArg::Input("top".to_owned())),
                ("b", InstructionArg::Input("bottom".to_owned())),
                ("k", InstructionArg::U64(0x00ff)),
                ("imm8", InstructionArg::U64(0x4e)),
            ],
        )],
        Vec::new(),
    );
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, &[(0, &input, &output)]);

    let asm = generate_solution_asm_for_paths(
        &metadata_for(ArchArg::Avx512, DTypeArg::I32),
        &table,
        &[path],
    );

    assert!(asm.contains("mov                  eax, 0xff"));
    assert!(asm.contains("kmovw                k1, eax"));
    assert!(asm.contains("zmm0{k1}"));
    assert!(asm.contains("k=0xff"));
}

#[test]
fn emits_shared_prefix_once_for_dual_side_gadgets() {
    let input = state(&[1, 3, 5, 7], &[2, 4, 6, 8]);
    let output = state(&[1, 2, 5, 6], &[3, 4, 7, 8]);
    let shared = inst(
        "_mm256_permute4x64_epi64",
        &[
            ("a", InstructionArg::Input("top".to_owned())),
            ("imm8", InstructionArg::U64(0x4e)),
        ],
    );
    let gadget = PermutationGadget::new(vec![shared.clone()], vec![shared]);
    let mut table = TransitionTable::new(1);
    table.add_transition(0, &input, &output, gadget);
    let path = path_for(&table, &[(0, &input, &output)]);

    let asm = generate_solution_asm_for_paths(&metadata(), &table, &[path]);

    assert!(asm.contains("; Shared instructions (1 of 1):"));
    assert_eq!(asm.matches("vpermq").count(), 1);
    assert!(asm.contains("; shared -> top"));
    assert!(asm.contains("; shared -> bottom"));
}

#[test]
fn emits_empty_asm_file_for_empty_path_set() {
    let table = TransitionTable::new(0);

    let asm = generate_solution_asm_for_paths(&metadata(), &table, &[]);

    assert!(asm.contains("section .text"));
    assert!(asm.contains("; no complete solutions exported"));
}
