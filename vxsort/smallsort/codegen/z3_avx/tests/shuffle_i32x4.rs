use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::control::{Imm8, mm_shuffle};
use z3_avx::registers::{construct_zmm_reg_from_elements, zmm_reg, zmm_reg_with_unique_values};
use z3_avx::shuffle_i32x4::{mm512_mask_shuffle_i32x4, mm512_shuffle_i32x4};

const NULL_SHUFFLE_I32X4_IMM8: u8 = 0xE4;

fn zmm_reg_pair_with_unique_values(prefix: &str, solver: &Solver) -> (BV, BV) {
    (
        zmm_reg_with_unique_values(&format!("{prefix}_a"), solver, 128),
        zmm_reg_with_unique_values(&format!("{prefix}_b"), solver, 128),
    )
}

#[test]
fn test_mm512_shuffle_i32x4_null_permute_works() {
    let solver = Solver::new();
    let input = zmm_reg("zmm0");
    let output = mm512_shuffle_i32x4(&input, &input, Imm8::Literal(NULL_SHUFFLE_I32X4_IMM8));

    solver.assert(output.eq(input).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_shuffle_i32x4_null_permute_found() {
    let solver = Solver::new();
    let input = zmm_reg_with_unique_values("zmm0", &solver, 128);
    let imm8 = BV::new_const("imm8", 8);
    let output = mm512_shuffle_i32x4(&input, &input, Imm8::Expr(imm8.clone()));

    solver.assert(output.eq(input));
    assert_eq!(solver.check(), SatResult::Sat);

    let model_imm8 = solver
        .get_model()
        .expect("sat model")
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm8, NULL_SHUFFLE_I32X4_IMM8 as u64);
}

#[test]
fn test_mm512_shuffle_i32x4_null_permute_2vec_works() {
    let solver = Solver::new();
    let (op1, op2) = zmm_reg_pair_with_unique_values("op", &solver);
    let output = mm512_shuffle_i32x4(&op1, &op2, Imm8::Literal(NULL_SHUFFLE_I32X4_IMM8));
    let expected =
        construct_zmm_reg_from_elements(128, &[(&op1, 0), (&op1, 1), (&op2, 2), (&op2, 3)]);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_shuffle_i32x4_null_permute_2vec_found() {
    let solver = Solver::new();
    let (op1, op2) = zmm_reg_pair_with_unique_values("op", &solver);
    let imm8 = BV::new_const("imm8_2vec", 8);
    let output = mm512_shuffle_i32x4(&op1, &op2, Imm8::Expr(imm8.clone()));
    let expected =
        construct_zmm_reg_from_elements(128, &[(&op1, 0), (&op1, 1), (&op2, 2), (&op2, 3)]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);

    let model_imm8 = solver
        .get_model()
        .expect("sat model")
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm8, NULL_SHUFFLE_I32X4_IMM8 as u64);
}

#[test]
fn test_mm512_shuffle_i32x4_cross_lanes() {
    let solver = Solver::new();
    let (a, b) = zmm_reg_pair_with_unique_values("input", &solver);
    let output = mm512_shuffle_i32x4(&a, &b, Imm8::Literal(mm_shuffle(0, 1, 2, 3)));
    let expected = construct_zmm_reg_from_elements(128, &[(&a, 3), (&a, 2), (&b, 1), (&b, 0)]);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mask_all_ones_equals_unmasked() {
    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("a", &solver, 32);
    let b = zmm_reg_with_unique_values("b", &solver, 32);
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let mask = BV::from_u64(0xFFFF, 16);

    let masked = mm512_mask_shuffle_i32x4(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_I32X4_IMM8),
        &solver,
    );
    let unmasked = mm512_shuffle_i32x4(&a, &b, Imm8::Literal(NULL_SHUFFLE_I32X4_IMM8));

    solver.assert(masked.eq(unmasked).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mask_all_zeros_preserves_src() {
    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("a", &solver, 32);
    let b = zmm_reg_with_unique_values("b", &solver, 32);
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let mask = BV::from_u64(0, 16);

    let result = mm512_mask_shuffle_i32x4(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_I32X4_IMM8),
        &solver,
    );

    solver.assert(result.eq(src).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_partial_mask() {
    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("a", &solver, 32);
    let b = zmm_reg_with_unique_values("b", &solver, 32);
    let src = zmm_reg_with_unique_values("src", &solver, 32);
    let mask = BV::from_u64(0x00FF, 16);

    let masked = mm512_mask_shuffle_i32x4(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_I32X4_IMM8),
        &solver,
    );
    let unmasked = mm512_shuffle_i32x4(&a, &b, Imm8::Literal(NULL_SHUFFLE_I32X4_IMM8));

    solver.assert(masked.extract(255, 0).eq(unmasked.extract(255, 0)).not());
    assert_eq!(solver.check(), SatResult::Unsat);

    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("a2", &solver, 32);
    let b = zmm_reg_with_unique_values("b2", &solver, 32);
    let src = zmm_reg_with_unique_values("src2", &solver, 32);
    let masked = mm512_mask_shuffle_i32x4(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_I32X4_IMM8),
        &solver,
    );

    solver.assert(masked.extract(511, 256).eq(src.extract(511, 256)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_shuffle_i32x4_canonicalizes_dead_mask_bits() {
    let solver = Solver::new();
    let src = zmm_reg("src");
    let a = zmm_reg("a");
    let b = zmm_reg("b");
    let mask = BV::new_const("wide_k", 64);

    let _result = mm512_mask_shuffle_i32x4(
        &src,
        &mask,
        &a,
        &b,
        Imm8::Literal(NULL_SHUFFLE_I32X4_IMM8),
        &solver,
    );

    solver.assert(mask.extract(63, 16).eq(BV::from_u64(0, 48)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}
