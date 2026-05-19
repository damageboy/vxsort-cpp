use gadget_synth::{
    Arch, DType, GadgetGraph, GadgetSynthesizer, InputRef, IntrinsicNode, Operand, Symbolic,
    SynthesisOptions, VectorState,
};

fn imm8_symbol(name: &str) -> Operand {
    Operand::Symbolic(Symbolic {
        name: name.to_owned(),
        bit_width: 8,
    })
}

fn input(name: &'static str) -> Operand {
    Operand::Input(InputRef { name })
}

fn permute_pd_node(input_name: &'static str, imm_name: &str) -> IntrinsicNode {
    IntrinsicNode::new("_mm256_permute_pd")
        .with_operand("a", input(input_name))
        .with_operand("imm8", imm8_symbol(imm_name))
}

fn permute2x128_node(input_name: &'static str, imm_name: &str) -> IntrinsicNode {
    IntrinsicNode::new("_mm256_permute2x128_si256")
        .with_operand("a", input(input_name))
        .with_operand("b", input(input_name))
        .with_operand("imm8", imm8_symbol(imm_name))
}

#[test]
fn synthesizes_avx2_i64_depth1_permute_pd_identity_for_both_sides() {
    let synth = GadgetSynthesizer::new(Arch::Avx2, DType::I64);
    let input = VectorState::new(vec![1, 2, 3, 4], vec![5, 6, 7, 8]);
    let target_pairs = vec![(1, 5), (2, 6), (3, 7), (4, 8)];
    let graph = GadgetGraph::new(
        Some(permute_pd_node("top", "test_identity_top")),
        Some(permute_pd_node("bottom", "test_identity_bottom")),
    );

    let results = synth
        .synthesize_graph(&graph, &input, &target_pairs, SynthesisOptions::default())
        .expect("synthesis should not error");

    assert_eq!(results.len(), 2);

    let (identity_gadget, identity_output) = &results[0];
    assert_eq!(identity_output.top(), &[1, 2, 3, 4]);
    assert_eq!(identity_output.bottom(), &[5, 6, 7, 8]);
    assert_eq!(identity_gadget.top_instructions().len(), 1);
    assert_eq!(identity_gadget.bottom_instructions().len(), 1);
    assert_eq!(
        identity_gadget.top_instructions()[0].intrinsic_name(),
        "_mm256_permute_pd"
    );
    assert_eq!(
        identity_gadget.top_instructions()[0].arg_u64("imm8"),
        Some(2)
    );
    assert_eq!(
        identity_gadget.bottom_instructions()[0].arg_u64("imm8"),
        Some(2)
    );

    let (swapped_gadget, swapped_output) = &results[1];
    assert_eq!(swapped_output.top(), &[2, 1, 4, 3]);
    assert_eq!(swapped_output.bottom(), &[6, 5, 8, 7]);
    assert_eq!(
        swapped_gadget.top_instructions()[0].arg_u64("imm8"),
        Some(1)
    );
    assert_eq!(
        swapped_gadget.bottom_instructions()[0].arg_u64("imm8"),
        Some(1)
    );
}

#[test]
fn synthesizes_avx2_i32_depth1_permute2x128_lane_swap_on_top_side() {
    let synth = GadgetSynthesizer::new(Arch::Avx2, DType::I32);
    let input_state = VectorState::new(
        vec![0, 1, 2, 3, 4, 5, 6, 7],
        vec![8, 9, 10, 11, 12, 13, 14, 15],
    );
    let target_pairs = vec![
        (4, 8),
        (5, 9),
        (6, 10),
        (7, 11),
        (0, 12),
        (1, 13),
        (2, 14),
        (3, 15),
    ];
    let graph = GadgetGraph::new(Some(permute2x128_node("top", "test_perm2x128_top")), None);

    let results = synth
        .synthesize_graph(
            &graph,
            &input_state,
            &target_pairs,
            SynthesisOptions::default(),
        )
        .expect("synthesis should not error");

    assert_eq!(results.len(), 1);
    let (gadget, output_state) = &results[0];
    assert_eq!(output_state.top(), &[4, 5, 6, 7, 0, 1, 2, 3]);
    assert_eq!(output_state.bottom(), &[8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(gadget.top_instructions().len(), 1);
    assert!(gadget.bottom_instructions().is_empty());
    assert_eq!(gadget.top_instructions()[0].arg_u64("imm8"), Some(1));
}
