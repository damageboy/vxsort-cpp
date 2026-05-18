use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::alignr::mm256_alignr_epi8;
use z3_avx::control::Imm8;
use z3_avx::registers::{construct_ymm_reg_from_elements, reg_with_unique_values};

fn ymm_reg_pair_with_unique_values(prefix: &str, solver: &Solver) -> (BV, BV) {
    (
        reg_with_unique_values(&format!("{prefix}_a"), solver, 32, 8),
        reg_with_unique_values(&format!("{prefix}_b"), solver, 32, 8),
    )
}

#[test]
fn test_mm256_alignr_epi8_shift_zero() {
    let solver = Solver::new();
    let (a, b) = ymm_reg_pair_with_unique_values("pair", &solver);
    let output = mm256_alignr_epi8(&a, &b, Imm8::Literal(0), &solver);
    let expected_specs: Vec<(&BV, usize)> = (0..32).map(|i| (&b, i)).collect();
    let expected = construct_ymm_reg_from_elements(8, &expected_specs);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm256_alignr_epi8_shift_one() {
    let solver = Solver::new();
    let (a, b) = ymm_reg_pair_with_unique_values("pair", &solver);
    let output = mm256_alignr_epi8(&a, &b, Imm8::Literal(1), &solver);
    let expected_specs: Vec<(&BV, usize)> = (1..16)
        .map(|i| (&b, i))
        .chain(std::iter::once((&a, 0)))
        .chain((17..32).map(|i| (&b, i)))
        .chain(std::iter::once((&a, 16)))
        .collect();
    let expected = construct_ymm_reg_from_elements(8, &expected_specs);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm256_alignr_epi8_shift_thirty_one() {
    let solver = Solver::new();
    let (a, b) = ymm_reg_pair_with_unique_values("pair", &solver);
    let zero = BV::from_u64(0, 8);
    let output = mm256_alignr_epi8(&a, &b, Imm8::Literal(31), &solver);
    let expected_specs: Vec<(&BV, usize)> = std::iter::once((&a, 15))
        .chain((0..15).map(|_| (&zero, 0)))
        .chain(std::iter::once((&a, 31)))
        .chain((0..15).map(|_| (&zero, 0)))
        .collect();
    let expected = construct_ymm_reg_from_elements(8, &expected_specs);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
}

#[test]
fn test_mm256_alignr_epi8_find_shift() {
    let solver = Solver::new();
    let (a, b) = ymm_reg_pair_with_unique_values("pair", &solver);
    let imm8 = BV::new_const("imm8_alignr_epi8", 8);
    let output = mm256_alignr_epi8(&a, &b, Imm8::Expr(imm8.clone()), &solver);
    let expected_specs: Vec<(&BV, usize)> = (2..16)
        .map(|i| (&b, i))
        .chain([(&a, 0), (&a, 1)])
        .chain((18..32).map(|i| (&b, i)))
        .chain([(&a, 16), (&a, 17)])
        .collect();
    let expected = construct_ymm_reg_from_elements(8, &expected_specs);

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
