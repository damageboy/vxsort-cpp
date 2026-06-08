use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::registers::{
    ymm_reg_with_64b_values, zmm_reg_with_64b_values, zmm_reg_with_unique_values,
};
use z3_avx::unpack::{
    mm256_unpackhi_epi64, mm256_unpacklo_epi64, mm512_mask_unpackhi_epi64,
    mm512_mask_unpacklo_epi64, mm512_unpackhi_epi64, mm512_unpacklo_epi64,
};

#[test]
fn test_mm256_unpacklo_epi64_interleaves_low_elements() {
    let solver = Solver::new();
    let a = ymm_reg_with_64b_values("a", &solver, &[10, 11, 12, 13]);
    let b = ymm_reg_with_64b_values("b", &solver, &[20, 21, 22, 23]);
    let result = mm256_unpacklo_epi64(&a, &b);
    let expected = ymm_reg_with_64b_values("exp", &solver, &[10, 20, 12, 22]);

    solver.assert(result.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm256_unpackhi_epi64_interleaves_high_elements() {
    let solver = Solver::new();
    let a = ymm_reg_with_64b_values("a", &solver, &[10, 11, 12, 13]);
    let b = ymm_reg_with_64b_values("b", &solver, &[20, 21, 22, 23]);
    let result = mm256_unpackhi_epi64(&a, &b);
    let expected = ymm_reg_with_64b_values("exp", &solver, &[11, 21, 13, 23]);

    solver.assert(result.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_unpacklo_epi64_interleaves_low_elements() {
    let solver = Solver::new();
    let a = zmm_reg_with_64b_values("a", &solver, &[10, 11, 12, 13, 14, 15, 16, 17]);
    let b = zmm_reg_with_64b_values("b", &solver, &[20, 21, 22, 23, 24, 25, 26, 27]);
    let result = mm512_unpacklo_epi64(&a, &b);
    let expected = zmm_reg_with_64b_values("exp", &solver, &[10, 20, 12, 22, 14, 24, 16, 26]);

    solver.assert(result.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_unpackhi_epi64_interleaves_high_elements() {
    let solver = Solver::new();
    let a = zmm_reg_with_64b_values("a", &solver, &[10, 11, 12, 13, 14, 15, 16, 17]);
    let b = zmm_reg_with_64b_values("b", &solver, &[20, 21, 22, 23, 24, 25, 26, 27]);
    let result = mm512_unpackhi_epi64(&a, &b);
    let expected = zmm_reg_with_64b_values("exp", &solver, &[11, 21, 13, 23, 15, 25, 17, 27]);

    solver.assert(result.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_mask_unpacklo_epi64_applies_writemask() {
    let solver = Solver::new();
    let src = zmm_reg_with_64b_values("src", &solver, &[100, 101, 102, 103, 104, 105, 106, 107]);
    let a = zmm_reg_with_64b_values("a", &solver, &[10, 11, 12, 13, 14, 15, 16, 17]);
    let b = zmm_reg_with_64b_values("b", &solver, &[20, 21, 22, 23, 24, 25, 26, 27]);
    let mask = BV::from_u64(0b1010_1010, 8);
    let result = mm512_mask_unpacklo_epi64(&src, &mask, &a, &b, &solver);
    let expected = zmm_reg_with_64b_values("exp", &solver, &[100, 20, 102, 22, 104, 24, 106, 26]);

    solver.assert(result.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_mask_unpackhi_epi64_applies_writemask() {
    let solver = Solver::new();
    let src = zmm_reg_with_64b_values("src", &solver, &[100, 101, 102, 103, 104, 105, 106, 107]);
    let a = zmm_reg_with_64b_values("a", &solver, &[10, 11, 12, 13, 14, 15, 16, 17]);
    let b = zmm_reg_with_64b_values("b", &solver, &[20, 21, 22, 23, 24, 25, 26, 27]);
    let mask = BV::from_u64(0b1111_0000, 8);
    let result = mm512_mask_unpackhi_epi64(&src, &mask, &a, &b, &solver);
    let expected = zmm_reg_with_64b_values("exp", &solver, &[100, 101, 102, 103, 15, 25, 17, 27]);

    solver.assert(result.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm512_mask_unpacklo_epi64_canonicalizes_dead_mask_bits() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src", &solver, 64);
    let a = zmm_reg_with_unique_values("a", &solver, 64);
    let b = zmm_reg_with_unique_values("b", &solver, 64);
    let mask = BV::new_const("wide_mask_epi64", 64);
    let output = mm512_mask_unpacklo_epi64(&src, &mask, &a, &b, &solver);
    solver.assert(output.eq(src));
    assert_eq!(solver.check(), SatResult::Sat);

    let model_mask = solver
        .get_model()
        .expect("sat model")
        .eval(&mask, true)
        .expect("mask model")
        .as_u64()
        .expect("mask numeral");
    assert_eq!(model_mask >> 8, 0);
}
