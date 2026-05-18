use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::blend::mm256_blend_pd;
use z3_avx::control::Imm8;
use z3_avx::registers::{construct_ymm_reg_from_elements, ymm_reg, ymm_reg_with_64b_values};

fn pair64(solver: &Solver) -> (BV, BV) {
    (
        ymm_reg_with_64b_values("a64", solver, &[1, 2, 3, 4]),
        ymm_reg_with_64b_values("b64", solver, &[101, 102, 103, 104]),
    )
}

#[test]
fn test_mm256_blend_pd_all_from_a() {
    let solver = Solver::new();
    let a = ymm_reg("a");
    let b = ymm_reg("b");
    let output = mm256_blend_pd(&a, &b, Imm8::Literal(0), &solver);

    solver.assert(output.eq(a).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blend_pd_all_from_b() {
    let solver = Solver::new();
    let a = ymm_reg("a");
    let b = ymm_reg("b");
    let output = mm256_blend_pd(&a, &b, Imm8::Literal(0b1111), &solver);

    solver.assert(output.eq(b).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blend_pd_alternating() {
    let solver = Solver::new();
    let (a, b) = pair64(&solver);
    let output = mm256_blend_pd(&a, &b, Imm8::Literal(0b1010), &solver);
    let expected = construct_ymm_reg_from_elements(64, &[(&a, 0), (&b, 1), (&a, 2), (&b, 3)]);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blend_pd_first_two_from_b() {
    let solver = Solver::new();
    let (a, b) = pair64(&solver);
    let output = mm256_blend_pd(&a, &b, Imm8::Literal(0b0011), &solver);
    let expected = construct_ymm_reg_from_elements(64, &[(&b, 0), (&b, 1), (&a, 2), (&a, 3)]);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blend_pd_symbolic_mask() {
    let solver = Solver::new();
    let (a, b) = pair64(&solver);
    let imm8 = BV::new_const("blend_pd_imm8", 8);
    let output = mm256_blend_pd(&a, &b, Imm8::Expr(imm8.clone()), &solver);
    let expected = construct_ymm_reg_from_elements(64, &[(&a, 0), (&b, 1), (&a, 2), (&b, 3)]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
    let model_imm8 = solver
        .get_model()
        .expect("sat model")
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm8 & 0xF, 0b1010);
}

#[test]
fn test_mm256_blend_pd_n3_stage_bottom_imm0() {
    let solver = Solver::new();
    let top = ymm_reg_with_64b_values("top", &solver, &[1, 3, 5, 7]);
    let bottom = ymm_reg_with_64b_values("bottom", &solver, &[2, 4, 6, 8]);
    let output = mm256_blend_pd(&top, &bottom, Imm8::Literal(0), &solver);
    let expected = ymm_reg_with_64b_values("expected", &solver, &[8, 6, 5, 7]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_blend_256_pd_canonicalization() {
    let solver = Solver::new();
    let a = ymm_reg("canon_a");
    let b = ymm_reg("canon_b");
    let imm8 = BV::new_const("canon_blend_pd_imm8", 8);

    let _result = mm256_blend_pd(&a, &b, Imm8::Expr(imm8.clone()), &solver);

    solver.assert(imm8.extract(7, 4).eq(BV::from_u64(0, 4)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}
