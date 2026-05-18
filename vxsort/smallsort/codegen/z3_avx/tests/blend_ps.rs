use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::blend::mm256_blend_ps;
use z3_avx::control::Imm8;
use z3_avx::registers::{construct_ymm_reg_from_elements, ymm_reg, ymm_reg_with_32b_values};

fn pair32(solver: &Solver) -> (BV, BV) {
    (
        ymm_reg_with_32b_values("a32", solver, &[1, 2, 3, 4, 5, 6, 7, 8]),
        ymm_reg_with_32b_values("b32", solver, &[101, 102, 103, 104, 105, 106, 107, 108]),
    )
}

#[test]
fn test_mm256_blend_ps_all_from_a() {
    let solver = Solver::new();
    let a = ymm_reg("a");
    let b = ymm_reg("b");
    let output = mm256_blend_ps(&a, &b, Imm8::Literal(0), &solver);

    solver.assert(output.eq(a).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blend_ps_all_from_b() {
    let solver = Solver::new();
    let a = ymm_reg("a");
    let b = ymm_reg("b");
    let output = mm256_blend_ps(&a, &b, Imm8::Literal(0xFF), &solver);

    solver.assert(output.eq(b).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blend_ps_alternating() {
    let solver = Solver::new();
    let (a, b) = pair32(&solver);
    let output = mm256_blend_ps(&a, &b, Imm8::Literal(0b10101010), &solver);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&a, 0),
            (&b, 1),
            (&a, 2),
            (&b, 3),
            (&a, 4),
            (&b, 5),
            (&a, 6),
            (&b, 7),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blend_ps_first_four_from_b() {
    let solver = Solver::new();
    let (a, b) = pair32(&solver);
    let output = mm256_blend_ps(&a, &b, Imm8::Literal(0b00001111), &solver);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&b, 0),
            (&b, 1),
            (&b, 2),
            (&b, 3),
            (&a, 4),
            (&a, 5),
            (&a, 6),
            (&a, 7),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blend_ps_symbolic_mask() {
    let solver = Solver::new();
    let (a, b) = pair32(&solver);
    let imm8 = BV::new_const("blend_ps_imm8", 8);
    let output = mm256_blend_ps(&a, &b, Imm8::Expr(imm8.clone()), &solver);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&b, 0),
            (&a, 1),
            (&b, 2),
            (&a, 3),
            (&b, 4),
            (&a, 5),
            (&b, 6),
            (&a, 7),
        ],
    );

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
    let model_imm8 = solver
        .get_model()
        .expect("sat model")
        .eval(&imm8, true)
        .expect("imm8 in model")
        .as_u64()
        .expect("imm8 numeral");
    assert_eq!(model_imm8, 0b01010101);
}
