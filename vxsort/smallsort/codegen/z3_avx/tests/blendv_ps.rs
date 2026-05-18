use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::blend::mm256_blendv_ps;
use z3_avx::registers::{construct_ymm_reg_from_elements, ymm_reg, ymm_reg_with_32b_values};

fn pair32(solver: &Solver) -> (BV, BV) {
    (
        ymm_reg_with_32b_values("a32v", solver, &[1, 2, 3, 4, 5, 6, 7, 8]),
        ymm_reg_with_32b_values("b32v", solver, &[101, 102, 103, 104, 105, 106, 107, 108]),
    )
}

fn assert_sign_bits(mask: &BV, solver: &Solver, bits: &[u64]) {
    for (element_idx, bit) in bits.iter().enumerate() {
        let high = element_idx as u32 * 32 + 31;
        solver.assert(mask.extract(high, high).eq(BV::from_u64(*bit, 1)));
    }
}

#[test]
fn test_mm256_blendv_ps_all_from_a() {
    let solver = Solver::new();
    let (a, b) = pair32(&solver);
    let mask = ymm_reg("mask_a");
    assert_sign_bits(&mask, &solver, &[0, 0, 0, 0, 0, 0, 0, 0]);

    let output = mm256_blendv_ps(&a, &b, &mask, &solver);

    solver.assert(output.eq(a).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blendv_ps_all_from_b() {
    let solver = Solver::new();
    let (a, b) = pair32(&solver);
    let mask = ymm_reg("mask_b");
    assert_sign_bits(&mask, &solver, &[1, 1, 1, 1, 1, 1, 1, 1]);

    let output = mm256_blendv_ps(&a, &b, &mask, &solver);

    solver.assert(output.eq(b).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blendv_ps_alternating() {
    let solver = Solver::new();
    let (a, b) = pair32(&solver);
    let mask = ymm_reg("mask_alt");
    assert_sign_bits(&mask, &solver, &[0, 1, 0, 1, 0, 1, 0, 1]);

    let output = mm256_blendv_ps(&a, &b, &mask, &solver);
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
fn test_mm256_blendv_ps_first_four_from_b() {
    let solver = Solver::new();
    let (a, b) = pair32(&solver);
    let mask = ymm_reg("mask_first_four");
    assert_sign_bits(&mask, &solver, &[1, 1, 1, 1, 0, 0, 0, 0]);

    let output = mm256_blendv_ps(&a, &b, &mask, &solver);
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
fn test_mm256_blendv_ps_symbolic_mask() {
    let solver = Solver::new();
    let (a, b) = pair32(&solver);
    let mask = ymm_reg("mask_symbolic");
    let output = mm256_blendv_ps(&a, &b, &mask, &solver);
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
    let model = solver.get_model().expect("sat model");
    for element_idx in 0..8 {
        let high = element_idx as u32 * 32 + 31;
        let bit = model
            .eval(&mask.extract(high, high), true)
            .expect("mask sign bit")
            .as_u64()
            .expect("mask sign bit numeral");
        let expected_bit = if element_idx % 2 == 0 { 1 } else { 0 };
        assert_eq!(bit, expected_bit);
    }
}
