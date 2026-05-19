use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::control::Imm8;
use z3_avx::lanes::extract_128b_lane;
use z3_avx::permute2x128::mm256_permute2x128_si256;
use z3_avx::registers::{construct_ymm_reg_from_elements, ymm_reg, ymm_reg_with_unique_values};

const NULL_PERMUTE2X128_IMM8: u8 = 0x10;

fn ymm_reg_pair_with_unique_values(prefix: &str, solver: &Solver) -> (BV, BV) {
    let a = ymm_reg_with_unique_values(&format!("{prefix}_a"), solver, 128);
    let b = ymm_reg_with_unique_values(&format!("{prefix}_b"), solver, 128);
    let lanes = [
        extract_128b_lane(&a, 0),
        extract_128b_lane(&a, 1),
        extract_128b_lane(&b, 0),
        extract_128b_lane(&b, 1),
    ];
    for i in 0..lanes.len() {
        for j in (i + 1)..lanes.len() {
            solver.assert(lanes[i].eq(&lanes[j]).not());
        }
    }
    (a, b)
}

#[test]
fn test_mm256_permute2x128_si256_null_permute_works() {
    let solver = Solver::new();
    let input = ymm_reg("ymm0");
    let output = mm256_permute2x128_si256(
        &input,
        &input,
        Imm8::Literal(NULL_PERMUTE2X128_IMM8),
        &solver,
    );

    solver.assert(output.eq(input).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute2x128_si256_null_permute_found() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 128);
    let imm8 = BV::new_const("imm8", 8);
    let output = mm256_permute2x128_si256(&input, &input, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(imm8.extract(3, 3).eq(BV::from_u64(0, 1)));
    solver.assert(imm8.extract(7, 7).eq(BV::from_u64(0, 1)));
    solver.assert(output.eq(input));
    assert_eq!(solver.check(), SatResult::Sat);

    let model_imm8 = solver
        .get_model()
        .expect("sat model")
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert!(
        [0x10, 0x12, 0x30, 0x32].contains(&model_imm8),
        "unexpected identity permute 0x{model_imm8:02x}"
    );
}

#[test]
fn test_mm256_permute2x128_si256_null_permute_2vec_works() {
    let solver = Solver::new();
    let (op1, op2) = ymm_reg_pair_with_unique_values("op", &solver);
    let output =
        mm256_permute2x128_si256(&op1, &op2, Imm8::Literal(NULL_PERMUTE2X128_IMM8), &solver);
    let expected = construct_ymm_reg_from_elements(128, &[(&op1, 0), (&op1, 1)]);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute2x128_si256_null_permute_2vec_found() {
    let solver = Solver::new();
    let (op1, op2) = ymm_reg_pair_with_unique_values("op", &solver);
    let imm8 = BV::new_const("imm8_2vec", 8);
    let output = mm256_permute2x128_si256(&op1, &op2, Imm8::Expr(imm8.clone()), &solver);
    let expected = construct_ymm_reg_from_elements(128, &[(&op1, 0), (&op1, 1)]);

    solver.assert(imm8.extract(3, 3).eq(BV::from_u64(0, 1)));
    solver.assert(imm8.extract(7, 7).eq(BV::from_u64(0, 1)));
    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);

    let model_imm8 = solver
        .get_model()
        .expect("sat model")
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm8, NULL_PERMUTE2X128_IMM8 as u64);
}

#[test]
fn test_mm256_permute2x128_si256_swap_lanes() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 128);
    let output = mm256_permute2x128_si256(&input, &input, Imm8::Literal(0x01), &solver);
    let expected = construct_ymm_reg_from_elements(128, &[(&input, 1), (&input, 0)]);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute2x128_si256_cross_vector() {
    let solver = Solver::new();
    let (a, b) = ymm_reg_pair_with_unique_values("input", &solver);
    let output = mm256_permute2x128_si256(&a, &b, Imm8::Literal(0x23), &solver);
    let expected = construct_ymm_reg_from_elements(128, &[(&b, 1), (&b, 0)]);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute2x128_si256_zero_lanes() {
    let solver = Solver::new();
    let input = ymm_reg_with_unique_values("ymm0", &solver, 128);
    let output = mm256_permute2x128_si256(&input, &input, Imm8::Literal(0x80), &solver);
    let expected = BV::from_u64(0, 128).concat(&input.extract(127, 0));

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permute2x128_si256_zero_both_lanes() {
    let solver = Solver::new();
    let input = ymm_reg("ymm0");
    let output = mm256_permute2x128_si256(&input, &input, Imm8::Literal(0x88), &solver);
    let expected = BV::from_u64(0, 256);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_permute2x128_canonicalization() {
    let solver = Solver::new();
    let a = ymm_reg("a");
    let b = ymm_reg("b");
    let imm8 = BV::new_const("canon_imm8", 8);

    let _result = mm256_permute2x128_si256(&a, &b, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(imm8.extract(2, 2).eq(BV::from_u64(0, 1)).not());
    assert_eq!(solver.check(), SatResult::Unsat);

    let solver = Solver::new();
    let imm8 = BV::new_const("canon_imm8_high", 8);
    let _result = mm256_permute2x128_si256(&a, &b, Imm8::Expr(imm8.clone()), &solver);
    solver.assert(imm8.extract(6, 6).eq(BV::from_u64(0, 1)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}
