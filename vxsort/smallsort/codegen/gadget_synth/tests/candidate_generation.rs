use gadget_synth::{Arch, DType, GadgetSynthesizer};

#[test]
fn avx2_i64_depth1_candidate_count_matches_python_menu() {
    let synth = GadgetSynthesizer::new(Arch::Avx2, DType::I64);

    let (shallow, deep) = synth
        .precompute_candidates_stratified(1)
        .expect("depth-1 candidate generation should succeed");

    assert_eq!(shallow.len(), 224);
    assert!(deep.is_empty());
}

#[test]
fn avx512_i32_depth1_candidate_count_matches_python_menu() {
    let synth = GadgetSynthesizer::new(Arch::Avx512, DType::I32);

    let (shallow, deep) = synth
        .precompute_candidates_stratified(1)
        .expect("depth-1 candidate generation should succeed");

    assert_eq!(shallow.len(), 575);
    assert!(deep.is_empty());
}

#[test]
fn avx2_i64_depth2_candidate_counts_match_python_menus() {
    let synth = GadgetSynthesizer::new(Arch::Avx2, DType::I64);

    let (shallow, deep) = synth
        .precompute_candidates_stratified(2)
        .expect("depth-2 candidate generation should succeed");
    let with_shared = synth
        .candidate_graph_templates(2)
        .expect("depth-2 candidate generation should succeed");
    let non_shared = synth
        .candidate_graph_templates_excluding_shared_prefix(2)
        .expect("depth-2 non-shared candidate generation should succeed");

    assert_eq!(shallow.len(), 224);
    assert_eq!(deep.len(), 8928);
    assert_eq!(with_shared.len(), 9152);
    assert_eq!(non_shared.len(), 9024);
}

#[test]
fn avx2_i64_depth2_candidate_tiers_split_shared_prefix_from_full_deep() {
    let synth = GadgetSynthesizer::new(Arch::Avx2, DType::I64);

    let tiers = synth
        .precompute_candidate_tiers(2)
        .expect("depth-2 tiered candidate generation should succeed");

    assert_eq!(tiers.shallow.len(), 224);
    assert_eq!(tiers.shared_prefix_deep.len(), 128);
    assert_eq!(tiers.full_deep.len(), 8800);
}

#[test]
fn candidate_graph_templates_rejects_depths_above_two() {
    let synth = GadgetSynthesizer::new(Arch::Avx2, DType::I64);

    let error = synth
        .candidate_graph_templates(3)
        .expect_err("depths above 2 should return an error");

    assert!(
        error
            .to_string()
            .contains("gadget depths above 2 are not supported yet"),
        "unexpected error: {error}"
    );
}
