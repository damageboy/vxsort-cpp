use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::control::Imm8;
use z3_avx::registers::{
    construct_ymm_reg_from_elements, construct_zmm_reg_from_elements, ymm_reg,
    ymm_reg_with_unique_values, zmm_reg, zmm_reg_with_unique_values,
};
use z3_avx::shuffle_pd::{mm256_shuffle_pd, mm512_mask_shuffle_pd, mm512_shuffle_pd};

const NULL_SHUFFLE_PD_AVX2_IMM8: u8 = 0x0A;
const NULL_SHUFFLE_PD_AVX512_IMM8: u8 = 0xAA;

#[test]
fn test_mm256_shuffle_pd_null_permute_works() {
    let solver = Solver::new();
    let input = ymm_reg("ymm0");
    let output = mm256_shuffle_pd(
        &input,
        &input,
        Imm8::Literal(NULL_SHUFFLE_PD_AVX2_IMM8),
        &solver,
    );

    solver.assert(output.eq(input).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_shuffle_pd_null_permute_found() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 64);
    let imm8 = BV::new_const("imm8_256", 8);
    let output = mm256_shuffle_pd(&input, &input, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(output.eq(input));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm, NULL_SHUFFLE_PD_AVX2_IMM8 as u64);
}

#[test]
fn test_mm256_shuffle_pd_null_permute_2vec_works() {
    let solver = Solver::new();
    let op1 = ymm_reg_with_unique_values("op1", &solver, 64);
    let op2 = ymm_reg_with_unique_values("op2", &solver, 64);
    let output = mm256_shuffle_pd(
        &op1,
        &op2,
        Imm8::Literal(NULL_SHUFFLE_PD_AVX2_IMM8),
        &solver,
    );
    let expected =
        construct_ymm_reg_from_elements(64, &[(&op1, 0), (&op2, 1), (&op1, 2), (&op2, 3)]);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_shuffle_pd_null_permute_2vec_found() {
    let solver = Solver::new();
    let op1 = ymm_reg_with_unique_values("op1", &solver, 64);
    let op2 = ymm_reg_with_unique_values("op2", &solver, 64);
    let imm8 = BV::new_const("imm8_256_2vec", 8);
    let output = mm256_shuffle_pd(&op1, &op2, Imm8::Expr(imm8.clone()), &solver);
    let expected =
        construct_ymm_reg_from_elements(64, &[(&op1, 0), (&op2, 1), (&op1, 2), (&op2, 3)]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm, NULL_SHUFFLE_PD_AVX2_IMM8 as u64);
}

#[test]
fn test_mm512_shuffle_pd_null_permute_works() {
    let solver = Solver::new();
    let input = zmm_reg("zmm0");
    let output = mm512_shuffle_pd(
        &input,
        &input,
        Imm8::Literal(NULL_SHUFFLE_PD_AVX512_IMM8),
        &solver,
    );

    solver.assert(output.eq(input).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_shuffle_pd_null_permute_found() {
    let solver = Solver::new();
    let input = zmm_reg_with_unique_values("zmm0", &solver, 64);
    let imm8 = BV::new_const("imm8_512", 8);
    let output = mm512_shuffle_pd(&input, &input, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(output.eq(input));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm, NULL_SHUFFLE_PD_AVX512_IMM8 as u64);
}

#[test]
fn test_mm512_shuffle_pd_null_permute_2vec_works() {
    let solver = Solver::new();
    let op1 = zmm_reg_with_unique_values("op1", &solver, 64);
    let op2 = zmm_reg_with_unique_values("op2", &solver, 64);
    let output = mm512_shuffle_pd(
        &op1,
        &op2,
        Imm8::Literal(NULL_SHUFFLE_PD_AVX512_IMM8),
        &solver,
    );
    let expected = construct_zmm_reg_from_elements(
        64,
        &[
            (&op1, 0),
            (&op2, 1),
            (&op1, 2),
            (&op2, 3),
            (&op1, 4),
            (&op2, 5),
            (&op1, 6),
            (&op2, 7),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_shuffle_pd_null_permute_2vec_found() {
    let solver = Solver::new();
    let op1 = zmm_reg_with_unique_values("op1", &solver, 64);
    let op2 = zmm_reg_with_unique_values("op2", &solver, 64);
    let imm8 = BV::new_const("imm8_512_2vec", 8);
    let output = mm512_shuffle_pd(&op1, &op2, Imm8::Expr(imm8.clone()), &solver);
    let expected = construct_zmm_reg_from_elements(
        64,
        &[
            (&op1, 0),
            (&op2, 1),
            (&op1, 2),
            (&op2, 3),
            (&op1, 4),
            (&op2, 5),
            (&op1, 6),
            (&op2, 7),
        ],
    );

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm, NULL_SHUFFLE_PD_AVX512_IMM8 as u64);
}

#[test]
fn test_mm512_mask_shuffle_pd_mask_all_zeros() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_zero_mask", &solver, 64);
    let a = zmm_reg_with_unique_values("a_zero_mask", &solver, 64);
    let b = zmm_reg_with_unique_values("b_zero_mask", &solver, 64);
    let mask = BV::from_u64(0, 8);

    let output = mm512_mask_shuffle_pd(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_PD_AVX512_IMM8),
        &solver,
    );

    solver.assert(output.eq(src).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_shuffle_pd_mask_all_ones() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_one_mask", &solver, 64);
    let a = zmm_reg_with_unique_values("a_one_mask", &solver, 64);
    let b = zmm_reg_with_unique_values("b_one_mask", &solver, 64);
    let mask = BV::from_u64(0xFF, 8);

    let masked = mm512_mask_shuffle_pd(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_PD_AVX512_IMM8),
        &solver,
    );
    let unmasked = mm512_shuffle_pd(&a, &b, Imm8::Literal(NULL_SHUFFLE_PD_AVX512_IMM8), &solver);

    solver.assert(masked.eq(unmasked).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_shuffle_pd_alternating_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_alt_mask", &solver, 64);
    let a = zmm_reg_with_unique_values("a_alt_mask", &solver, 64);
    let b = zmm_reg_with_unique_values("b_alt_mask", &solver, 64);
    let mask = BV::from_u64(0x55, 8);

    let output = mm512_mask_shuffle_pd(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_PD_AVX512_IMM8),
        &solver,
    );
    let unmasked = mm512_shuffle_pd(&a, &b, Imm8::Literal(NULL_SHUFFLE_PD_AVX512_IMM8), &solver);
    let expected_specs: Vec<(&BV, usize)> = (0..8)
        .map(|i| {
            if i % 2 == 0 {
                (&unmasked, i)
            } else {
                (&src, i)
            }
        })
        .collect();
    let expected = construct_zmm_reg_from_elements(64, &expected_specs);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_shuffle_pd_256_canonicalization() {
    let solver = Solver::new();
    let a = ymm_reg("canon_a");
    let b = ymm_reg("canon_b");
    let imm8 = BV::new_const("canon_imm8", 8);

    let _output = mm256_shuffle_pd(&a, &b, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(imm8.extract(7, 4).eq(BV::from_u64(0, 4)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_shuffle_pd_canonicalizes_dead_mask_bits() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_wide_mask", &solver, 64);
    let a = zmm_reg_with_unique_values("a_wide_mask", &solver, 64);
    let b = zmm_reg_with_unique_values("b_wide_mask", &solver, 64);
    let mask = BV::new_const("k_wide_pd", 64);

    let _output = mm512_mask_shuffle_pd(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_PD_AVX512_IMM8),
        &solver,
    );

    solver.assert(mask.extract(63, 8).eq(BV::from_u64(0, 56)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}
