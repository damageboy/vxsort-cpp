use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::alignr::{mm256_alignr_epi64, mm512_alignr_epi64, mm512_mask_alignr_epi64};
use z3_avx::control::Imm8;
use z3_avx::registers::{ymm_reg_with_64b_values, zmm_reg_with_64b_values};

#[test]
fn test_mm256_alignr_epi64_shift_zero() {
    let solver = Solver::new();
    let a = ymm_reg_with_64b_values("a", &solver, &(100..104).collect::<Vec<_>>());
    let b = ymm_reg_with_64b_values("b", &solver, &(0..4).collect::<Vec<_>>());
    let output = mm256_alignr_epi64(&a, &b, Imm8::Literal(0), &solver);
    let expected = ymm_reg_with_64b_values("expected", &solver, &(0..4).collect::<Vec<_>>());

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm256_alignr_epi64_shift_one() {
    let solver = Solver::new();
    let a = ymm_reg_with_64b_values("a", &solver, &(100..104).collect::<Vec<_>>());
    let b = ymm_reg_with_64b_values("b", &solver, &(0..4).collect::<Vec<_>>());
    let output = mm256_alignr_epi64(&a, &b, Imm8::Literal(1), &solver);
    let expected = ymm_reg_with_64b_values("expected", &solver, &[1, 2, 3, 100]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm256_alignr_epi64_shift_two() {
    let solver = Solver::new();
    let a = ymm_reg_with_64b_values("a", &solver, &(100..104).collect::<Vec<_>>());
    let b = ymm_reg_with_64b_values("b", &solver, &(0..4).collect::<Vec<_>>());
    let output = mm256_alignr_epi64(&a, &b, Imm8::Literal(2), &solver);
    let expected = ymm_reg_with_64b_values("expected", &solver, &[2, 3, 100, 101]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm256_alignr_epi64_shift_three() {
    let solver = Solver::new();
    let a = ymm_reg_with_64b_values("a", &solver, &(100..104).collect::<Vec<_>>());
    let b = ymm_reg_with_64b_values("b", &solver, &(0..4).collect::<Vec<_>>());
    let output = mm256_alignr_epi64(&a, &b, Imm8::Literal(3), &solver);
    let expected = ymm_reg_with_64b_values("expected", &solver, &[3, 100, 101, 102]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_alignr_epi64_shift_zero() {
    let solver = Solver::new();
    let a = zmm_reg_with_64b_values("a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let output = mm512_alignr_epi64(&a, &b, Imm8::Literal(0), &solver);
    let expected = zmm_reg_with_64b_values("expected", &solver, &(0..8).collect::<Vec<_>>());

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_alignr_epi64_shift_seven() {
    let solver = Solver::new();
    let a = zmm_reg_with_64b_values("a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let output = mm512_alignr_epi64(&a, &b, Imm8::Literal(7), &solver);
    let expected =
        zmm_reg_with_64b_values("expected", &solver, &[7, 200, 201, 202, 203, 204, 205, 206]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_alignr_epi64_shift_four() {
    let solver = Solver::new();
    let a = zmm_reg_with_64b_values("a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let output = mm512_alignr_epi64(&a, &b, Imm8::Literal(4), &solver);
    let expected = zmm_reg_with_64b_values("expected", &solver, &[4, 5, 6, 7, 200, 201, 202, 203]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_alignr_epi64_shift_three() {
    let solver = Solver::new();
    let a = zmm_reg_with_64b_values("a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let output = mm512_alignr_epi64(&a, &b, Imm8::Literal(3), &solver);
    let expected = zmm_reg_with_64b_values("expected", &solver, &[3, 4, 5, 6, 7, 200, 201, 202]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm256_alignr_epi64_find_shift() {
    let solver = Solver::new();
    let a = ymm_reg_with_64b_values("a", &solver, &(100..104).collect::<Vec<_>>());
    let b = ymm_reg_with_64b_values("b", &solver, &(0..4).collect::<Vec<_>>());
    let imm8 = BV::new_const("imm8", 8);
    let output = mm256_alignr_epi64(&a, &b, Imm8::Expr(imm8.clone()), &solver);
    let expected = ymm_reg_with_64b_values("expected", &solver, &[1, 2, 3, 100]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
    let model_imm8 = solver
        .get_model()
        .expect("sat model")
        .eval(&imm8, true)
        .expect("imm model")
        .as_u64()
        .expect("imm numeral");
    assert_eq!(model_imm8, 1);
}

#[test]
fn test_mm256_alignr_epi64_canonicalizes_imm_high_bits() {
    let solver = Solver::new();
    let a = ymm_reg_with_64b_values("a", &solver, &(100..104).collect::<Vec<_>>());
    let b = ymm_reg_with_64b_values("b", &solver, &(0..4).collect::<Vec<_>>());
    let imm8 = BV::new_const("imm8_256_epi64", 8);
    let _output = mm256_alignr_epi64(&a, &b, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(imm8.extract(7, 2).eq(BV::from_u64(0, 6)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_alignr_epi64_canonicalizes_imm_high_bits() {
    let solver = Solver::new();
    let a = zmm_reg_with_64b_values("a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let imm8 = BV::new_const("imm8_512_epi64", 8);
    let _output = mm512_alignr_epi64(&a, &b, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(imm8.extract(7, 3).eq(BV::from_u64(0, 5)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_alignr_epi64_mask_all_zeros() {
    let solver = Solver::new();
    let src = zmm_reg_with_64b_values("src", &solver, &(100..108).collect::<Vec<_>>());
    let a = zmm_reg_with_64b_values("a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let k = BV::from_u64(0x00, 8);
    let output = mm512_mask_alignr_epi64(&src, &k, &a, &b, Imm8::Literal(2), &solver);

    solver.assert(output.eq(src));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_mask_alignr_epi64_mask_all_ones() {
    let solver = Solver::new();
    let src = zmm_reg_with_64b_values("src", &solver, &(100..108).collect::<Vec<_>>());
    let a = zmm_reg_with_64b_values("a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let k = BV::from_u64(0xFF, 8);
    let output = mm512_mask_alignr_epi64(&src, &k, &a, &b, Imm8::Literal(2), &solver);
    let expected = zmm_reg_with_64b_values("expected", &solver, &[2, 3, 4, 5, 6, 7, 200, 201]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_mask_alignr_epi64_alternating_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_64b_values("src", &solver, &(100..108).collect::<Vec<_>>());
    let a = zmm_reg_with_64b_values("a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let k = BV::from_u64(0xAA, 8);
    let output = mm512_mask_alignr_epi64(&src, &k, &a, &b, Imm8::Literal(1), &solver);
    let expected =
        zmm_reg_with_64b_values("expected", &solver, &[100, 2, 102, 4, 104, 6, 106, 200]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_mask_alignr_epi64_partial_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_64b_values("src", &solver, &(100..108).collect::<Vec<_>>());
    let a = zmm_reg_with_64b_values("a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let k = BV::from_u64(0x0F, 8);
    let output = mm512_mask_alignr_epi64(&src, &k, &a, &b, Imm8::Literal(3), &solver);
    let expected = zmm_reg_with_64b_values("expected", &solver, &[3, 4, 5, 6, 104, 105, 106, 107]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_mask_alignr_epi64_single_bit_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_64b_values("src", &solver, &(100..108).collect::<Vec<_>>());
    let a = zmm_reg_with_64b_values("a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let k = BV::from_u64(0x10, 8);
    let output = mm512_mask_alignr_epi64(&src, &k, &a, &b, Imm8::Literal(2), &solver);
    let expected =
        zmm_reg_with_64b_values("expected", &solver, &[100, 101, 102, 103, 6, 105, 106, 107]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_mask_alignr_epi64_find_shift_and_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_64b_values("src", &solver, &(100..108).collect::<Vec<_>>());
    let a = zmm_reg_with_64b_values("a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let k = BV::new_const("k", 8);
    let imm8 = BV::new_const("imm8", 8);
    let output = mm512_mask_alignr_epi64(&src, &k, &a, &b, Imm8::Expr(imm8.clone()), &solver);
    let expected =
        zmm_reg_with_64b_values("expected", &solver, &[5, 6, 7, 103, 104, 105, 106, 107]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
    let model = solver.get_model().expect("sat model");
    let model_imm8 = model
        .eval(&imm8, true)
        .expect("imm model")
        .as_u64()
        .expect("imm numeral");
    let model_k = model
        .eval(&k, true)
        .expect("mask model")
        .as_u64()
        .expect("mask numeral");
    assert_eq!(model_imm8, 5);
    assert_eq!(model_k, 0x07);
}

#[test]
fn test_mm512_mask_alignr_epi64_canonicalizes_dead_mask_bits() {
    let solver = Solver::new();
    let src = zmm_reg_with_64b_values("src", &solver, &(100..108).collect::<Vec<_>>());
    let a = zmm_reg_with_64b_values("a", &solver, &(200..208).collect::<Vec<_>>());
    let b = zmm_reg_with_64b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let k = BV::new_const("wide_k_epi64", 64);
    let output = mm512_mask_alignr_epi64(&src, &k, &a, &b, Imm8::Literal(2), &solver);

    solver.assert(output.eq(src));
    assert_eq!(solver.check(), SatResult::Sat);
    let model_k = solver
        .get_model()
        .expect("sat model")
        .eval(&k, true)
        .expect("mask model")
        .as_u64()
        .expect("mask numeral");
    assert_eq!(model_k >> 8, 0);
}
