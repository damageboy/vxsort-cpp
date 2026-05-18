use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::alignr::{mm256_alignr_epi32, mm512_alignr_epi32, mm512_mask_alignr_epi32};
use z3_avx::control::Imm8;
use z3_avx::registers::{ymm_reg_with_32b_values, zmm_reg_with_32b_values};

#[test]
fn test_mm256_alignr_epi32_shift_zero() {
    let solver = Solver::new();
    let a = ymm_reg_with_32b_values("a", &solver, &(10..18).collect::<Vec<_>>());
    let b = ymm_reg_with_32b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let output = mm256_alignr_epi32(&a, &b, Imm8::Literal(0), &solver);
    let expected = ymm_reg_with_32b_values("expected", &solver, &(0..8).collect::<Vec<_>>());

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm256_alignr_epi32_shift_one() {
    let solver = Solver::new();
    let a = ymm_reg_with_32b_values("a", &solver, &(10..18).collect::<Vec<_>>());
    let b = ymm_reg_with_32b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let output = mm256_alignr_epi32(&a, &b, Imm8::Literal(1), &solver);
    let expected = ymm_reg_with_32b_values("expected", &solver, &[1, 2, 3, 4, 5, 6, 7, 10]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm256_alignr_epi32_shift_seven() {
    let solver = Solver::new();
    let a = ymm_reg_with_32b_values("a", &solver, &(10..18).collect::<Vec<_>>());
    let b = ymm_reg_with_32b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let output = mm256_alignr_epi32(&a, &b, Imm8::Literal(7), &solver);
    let expected = ymm_reg_with_32b_values("expected", &solver, &[7, 10, 11, 12, 13, 14, 15, 16]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm256_alignr_epi32_shift_four() {
    let solver = Solver::new();
    let a = ymm_reg_with_32b_values("a", &solver, &(10..18).collect::<Vec<_>>());
    let b = ymm_reg_with_32b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let output = mm256_alignr_epi32(&a, &b, Imm8::Literal(4), &solver);
    let expected = ymm_reg_with_32b_values("expected", &solver, &[4, 5, 6, 7, 10, 11, 12, 13]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_alignr_epi32_shift_zero() {
    let solver = Solver::new();
    let a = zmm_reg_with_32b_values("a", &solver, &(20..36).collect::<Vec<_>>());
    let b = zmm_reg_with_32b_values("b", &solver, &(0..16).collect::<Vec<_>>());
    let output = mm512_alignr_epi32(&a, &b, Imm8::Literal(0), &solver);
    let expected = zmm_reg_with_32b_values("expected", &solver, &(0..16).collect::<Vec<_>>());

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_alignr_epi32_shift_fifteen() {
    let solver = Solver::new();
    let a = zmm_reg_with_32b_values("a", &solver, &(20..36).collect::<Vec<_>>());
    let b = zmm_reg_with_32b_values("b", &solver, &(0..16).collect::<Vec<_>>());
    let output = mm512_alignr_epi32(&a, &b, Imm8::Literal(15), &solver);
    let expected_values: Vec<u64> = std::iter::once(15).chain(20..35).collect();
    let expected = zmm_reg_with_32b_values("expected", &solver, &expected_values);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_alignr_epi32_shift_four() {
    let solver = Solver::new();
    let a = zmm_reg_with_32b_values("a", &solver, &(16..32).collect::<Vec<_>>());
    let b = zmm_reg_with_32b_values("b", &solver, &(0..16).collect::<Vec<_>>());
    let output = mm512_alignr_epi32(&a, &b, Imm8::Literal(4), &solver);
    let expected = zmm_reg_with_32b_values("expected", &solver, &(4..20).collect::<Vec<_>>());

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm256_alignr_epi32_find_shift() {
    let solver = Solver::new();
    let a = ymm_reg_with_32b_values("a", &solver, &(10..18).collect::<Vec<_>>());
    let b = ymm_reg_with_32b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let imm8 = BV::new_const("imm8", 8);
    let output = mm256_alignr_epi32(&a, &b, Imm8::Expr(imm8.clone()), &solver);
    let expected = ymm_reg_with_32b_values("expected", &solver, &[2, 3, 4, 5, 6, 7, 10, 11]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
    let model_imm8 = solver
        .get_model()
        .expect("sat model")
        .eval(&imm8, true)
        .expect("imm model")
        .as_u64()
        .expect("imm numeral");
    assert_eq!(model_imm8, 2);
}

#[test]
fn test_mm256_alignr_epi32_canonicalizes_imm_high_bits() {
    let solver = Solver::new();
    let a = ymm_reg_with_32b_values("a", &solver, &(10..18).collect::<Vec<_>>());
    let b = ymm_reg_with_32b_values("b", &solver, &(0..8).collect::<Vec<_>>());
    let imm8 = BV::new_const("imm8_256_epi32", 8);
    let _output = mm256_alignr_epi32(&a, &b, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(imm8.extract(7, 3).eq(BV::from_u64(0, 5)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_alignr_epi32_canonicalizes_imm_high_bits() {
    let solver = Solver::new();
    let a = zmm_reg_with_32b_values("a", &solver, &(20..36).collect::<Vec<_>>());
    let b = zmm_reg_with_32b_values("b", &solver, &(0..16).collect::<Vec<_>>());
    let imm8 = BV::new_const("imm8_512_epi32", 8);
    let _output = mm512_alignr_epi32(&a, &b, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(imm8.extract(7, 4).eq(BV::from_u64(0, 4)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_alignr_epi32_mask_all_zeros() {
    let solver = Solver::new();
    let src = zmm_reg_with_32b_values("src", &solver, &(100..116).collect::<Vec<_>>());
    let a = zmm_reg_with_32b_values("a", &solver, &(20..36).collect::<Vec<_>>());
    let b = zmm_reg_with_32b_values("b", &solver, &(0..16).collect::<Vec<_>>());
    let k = BV::from_u64(0x0000, 16);
    let output = mm512_mask_alignr_epi32(&src, &k, &a, &b, Imm8::Literal(4), &solver);

    solver.assert(output.eq(src));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_mask_alignr_epi32_mask_all_ones() {
    let solver = Solver::new();
    let src = zmm_reg_with_32b_values("src", &solver, &(100..116).collect::<Vec<_>>());
    let a = zmm_reg_with_32b_values("a", &solver, &(20..36).collect::<Vec<_>>());
    let b = zmm_reg_with_32b_values("b", &solver, &(0..16).collect::<Vec<_>>());
    let k = BV::from_u64(0xFFFF, 16);
    let output = mm512_mask_alignr_epi32(&src, &k, &a, &b, Imm8::Literal(4), &solver);
    let expected = zmm_reg_with_32b_values(
        "expected",
        &solver,
        &[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23],
    );

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_mask_alignr_epi32_alternating_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_32b_values("src", &solver, &(100..116).collect::<Vec<_>>());
    let a = zmm_reg_with_32b_values("a", &solver, &(20..36).collect::<Vec<_>>());
    let b = zmm_reg_with_32b_values("b", &solver, &(0..16).collect::<Vec<_>>());
    let k = BV::from_u64(0xAAAA, 16);
    let output = mm512_mask_alignr_epi32(&src, &k, &a, &b, Imm8::Literal(2), &solver);
    let expected = zmm_reg_with_32b_values(
        "expected",
        &solver,
        &[
            100, 3, 102, 5, 104, 7, 106, 9, 108, 11, 110, 13, 112, 15, 114, 21,
        ],
    );

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_mask_alignr_epi32_partial_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_32b_values("src", &solver, &(100..116).collect::<Vec<_>>());
    let a = zmm_reg_with_32b_values("a", &solver, &(20..36).collect::<Vec<_>>());
    let b = zmm_reg_with_32b_values("b", &solver, &(0..16).collect::<Vec<_>>());
    let k = BV::from_u64(0x00FF, 16);
    let output = mm512_mask_alignr_epi32(&src, &k, &a, &b, Imm8::Literal(1), &solver);
    let expected = zmm_reg_with_32b_values(
        "expected",
        &solver,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 108, 109, 110, 111, 112, 113, 114, 115,
        ],
    );

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_mask_alignr_epi32_find_mask() {
    let solver = Solver::new();
    let src = zmm_reg_with_32b_values("src", &solver, &(100..116).collect::<Vec<_>>());
    let a = zmm_reg_with_32b_values("a", &solver, &(20..36).collect::<Vec<_>>());
    let b = zmm_reg_with_32b_values("b", &solver, &(0..16).collect::<Vec<_>>());
    let k = BV::new_const("k", 16);
    let output = mm512_mask_alignr_epi32(&src, &k, &a, &b, Imm8::Literal(8), &solver);
    let expected = zmm_reg_with_32b_values(
        "expected",
        &solver,
        &[
            8, 101, 10, 103, 12, 105, 14, 107, 20, 109, 22, 111, 24, 113, 26, 115,
        ],
    );

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
    let model_k = solver
        .get_model()
        .expect("sat model")
        .eval(&k, true)
        .expect("mask model")
        .as_u64()
        .expect("mask numeral");
    assert_eq!(model_k, 0x5555);
}

#[test]
fn test_mm512_mask_alignr_epi32_canonicalizes_dead_mask_bits() {
    let solver = Solver::new();
    let src = zmm_reg_with_32b_values("src", &solver, &(100..116).collect::<Vec<_>>());
    let a = zmm_reg_with_32b_values("a", &solver, &(20..36).collect::<Vec<_>>());
    let b = zmm_reg_with_32b_values("b", &solver, &(0..16).collect::<Vec<_>>());
    let k = BV::new_const("wide_k_epi32", 64);
    let output = mm512_mask_alignr_epi32(&src, &k, &a, &b, Imm8::Literal(4), &solver);

    solver.assert(output.eq(src));
    assert_eq!(solver.check(), SatResult::Sat);
    let model_k = solver
        .get_model()
        .expect("sat model")
        .eval(&k, true)
        .expect("mask model")
        .as_u64()
        .expect("mask numeral");
    assert_eq!(model_k >> 16, 0);
}
