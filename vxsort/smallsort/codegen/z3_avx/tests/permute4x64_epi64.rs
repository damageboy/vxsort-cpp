use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::control::{Imm8, mm_shuffle};
use z3_avx::permute4x64::mm256_permute4x64_epi64;
use z3_avx::registers::{
    construct_ymm_reg_from_elements, ymm_reg, ymm_reg_reversed, ymm_reg_with_unique_values,
};

#[test]
fn test_mm256_permute4x64_epi64_identity() {
    let solver = Solver::new();
    let input = ymm_reg("ymm0");
    let imm8 = mm_shuffle(3, 2, 1, 0);

    let output = mm256_permute4x64_epi64(&input, Imm8::Literal(imm8));

    solver.assert(output.eq(&input).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute4x64_epi64_reverse() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 64);
    let imm8 = mm_shuffle(0, 1, 2, 3);

    let output = mm256_permute4x64_epi64(&input, Imm8::Literal(imm8));
    let reversed_input = ymm_reg_reversed("ymm_reversed", &solver, &input, 64);

    solver.assert(output.eq(&reversed_input).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute4x64_epi64_broadcast_first() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 64);
    let imm8 = mm_shuffle(0, 0, 0, 0);

    let output = mm256_permute4x64_epi64(&input, Imm8::Literal(imm8));
    let expected =
        construct_ymm_reg_from_elements(64, &[(&input, 0), (&input, 0), (&input, 0), (&input, 0)]);

    solver.assert(output.eq(&expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute4x64_epi64_broadcast_last() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 64);
    let imm8 = mm_shuffle(3, 3, 3, 3);

    let output = mm256_permute4x64_epi64(&input, Imm8::Literal(imm8));
    let expected =
        construct_ymm_reg_from_elements(64, &[(&input, 3), (&input, 3), (&input, 3), (&input, 3)]);

    solver.assert(output.eq(&expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute4x64_epi64_swap_pairs() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 64);
    let imm8 = mm_shuffle(2, 3, 0, 1);

    let output = mm256_permute4x64_epi64(&input, Imm8::Literal(imm8));
    let expected =
        construct_ymm_reg_from_elements(64, &[(&input, 1), (&input, 0), (&input, 3), (&input, 2)]);

    solver.assert(output.eq(&expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute4x64_epi64_swap_halves() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 64);
    let imm8 = mm_shuffle(1, 0, 3, 2);

    let output = mm256_permute4x64_epi64(&input, Imm8::Literal(imm8));
    let expected =
        construct_ymm_reg_from_elements(64, &[(&input, 2), (&input, 3), (&input, 0), (&input, 1)]);

    solver.assert(output.eq(&expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute4x64_epi64_custom_pattern() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 64);
    let imm8 = mm_shuffle(0, 2, 3, 1);

    let output = mm256_permute4x64_epi64(&input, Imm8::Literal(imm8));
    let expected =
        construct_ymm_reg_from_elements(64, &[(&input, 1), (&input, 3), (&input, 2), (&input, 0)]);

    solver.assert(output.eq(&expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute4x64_epi64_symbolic_imm() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 64);
    let imm8 = BV::new_const("imm8", 8);

    let output = mm256_permute4x64_epi64(&input, Imm8::Expr(imm8.clone()));
    let expected =
        construct_ymm_reg_from_elements(64, &[(&input, 2), (&input, 0), (&input, 3), (&input, 1)]);

    solver.assert(output.eq(&expected));
    assert_eq!(solver.check(), SatResult::Sat);

    let model = solver.get_model().expect("sat model");
    let model_imm8 = model
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm8, mm_shuffle(1, 3, 0, 2) as u64);
}
