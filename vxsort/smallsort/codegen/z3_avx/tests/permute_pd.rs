use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::control::{Imm8, mm_shuffle2};
use z3_avx::permute_pd::{mm256_permute_pd, mm512_mask_permute_pd, mm512_permute_pd};
use z3_avx::registers::{
    construct_zmm_reg_from_elements, ymm_reg, ymm_reg_with_unique_values, zmm_reg,
    zmm_reg_with_unique_values,
};

const NULL_PERMUTE_PD_IMM8: u8 = 0b10;

#[test]
fn test_mm256_permute_epi64_null_permute_works() {
    let solver = Solver::new();
    let input = ymm_reg("ymm0");
    let output = mm256_permute_pd(&input, Imm8::Literal(NULL_PERMUTE_PD_IMM8), &solver);

    solver.assert(input.eq(output).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute_epi64_null_permute_found() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 64);
    let imm8 = BV::new_const("imm8_256", 8);
    let output = mm256_permute_pd(&input, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(input.eq(output));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm8 = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm8, NULL_PERMUTE_PD_IMM8 as u64);
}

#[test]
fn test_mm512_permute_epi64_null_permute_works() {
    let solver = Solver::new();
    let input = zmm_reg("zmm0");
    let output = mm512_permute_pd(&input, Imm8::Literal(NULL_PERMUTE_PD_IMM8), &solver);

    solver.assert(input.eq(output).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_permute_epi64_null_permute_found() {
    let solver = Solver::new();
    let input = zmm_reg_with_unique_values("zmm0", &solver, 64);
    let imm8 = BV::new_const("imm8_512", 8);
    let output = mm512_permute_pd(&input, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(input.eq(output));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm8 = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm8, NULL_PERMUTE_PD_IMM8 as u64);
}

#[test]
fn test_mm512_mask_permute_pd_mask_all_zeros() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_zero_mask", &solver, 64);
    let a = zmm_reg_with_unique_values("a_zero_mask", &solver, 64);
    let mask = BV::from_u64(0, 8);

    let output = mm512_mask_permute_pd(
        &src,
        &mask,
        &a,
        Imm8::Literal(NULL_PERMUTE_PD_IMM8),
        &solver,
    );

    solver.assert(output.eq(src).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permute_pd_mask_all_ones() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_one_mask", &solver, 64);
    let a = zmm_reg_with_unique_values("a_one_mask", &solver, 64);
    let mask = BV::from_u64(0xFF, 8);

    let masked_output = mm512_mask_permute_pd(
        &src,
        &mask,
        &a,
        Imm8::Literal(NULL_PERMUTE_PD_IMM8),
        &solver,
    );
    let unmasked_output = mm512_permute_pd(&a, Imm8::Literal(NULL_PERMUTE_PD_IMM8), &solver);

    solver.assert(masked_output.eq(unmasked_output).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permute_pd_single_bit_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_single_bit_mask", &solver, 64);
    let a = zmm_reg_with_unique_values("a_single_bit_mask", &solver, 64);
    let mask = BV::from_u64(1 << 3, 8);
    let imm8 = mm_shuffle2(0, 1);

    let output = mm512_mask_permute_pd(&src, &mask, &a, Imm8::Literal(imm8), &solver);
    let unmasked = mm512_permute_pd(&a, Imm8::Literal(imm8), &solver);

    let expected_specs: Vec<(&BV, usize)> = (0..8)
        .map(|i| if i == 3 { (&unmasked, i) } else { (&src, i) })
        .collect();
    let expected = construct_zmm_reg_from_elements(64, &expected_specs);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_permute_pd_256_canonicalization() {
    let solver = Solver::new();
    let a = ymm_reg("a_canonical");
    let imm8 = BV::new_const("imm8_canonical", 8);

    let _output = mm256_permute_pd(&a, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(imm8.extract(7, 2).eq(BV::from_u64(0, 6)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permute_pd_canonicalizes_dead_mask_bits() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_wide_mask", &solver, 64);
    let a = zmm_reg_with_unique_values("a_wide_mask", &solver, 64);
    let mask = BV::new_const("k_wide", 64);

    let _output = mm512_mask_permute_pd(
        &src,
        &mask,
        &a,
        Imm8::Literal(NULL_PERMUTE_PD_IMM8),
        &solver,
    );

    solver.assert(mask.extract(63, 8).eq(BV::from_u64(0, 56)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}
