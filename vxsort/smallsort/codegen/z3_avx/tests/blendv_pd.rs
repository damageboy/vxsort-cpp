use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::blend::mm256_blendv_pd;
use z3_avx::registers::{construct_ymm_reg_from_elements, ymm_reg, ymm_reg_with_64b_values};

fn pair64(solver: &Solver) -> (BV, BV) {
    (
        ymm_reg_with_64b_values("a64v", solver, &[1, 2, 3, 4]),
        ymm_reg_with_64b_values("b64v", solver, &[101, 102, 103, 104]),
    )
}

fn assert_sign_bits(mask: &BV, solver: &Solver, bits: &[u64]) {
    for (element_idx, bit) in bits.iter().enumerate() {
        let high = element_idx as u32 * 64 + 63;
        solver.assert(mask.extract(high, high).eq(BV::from_u64(*bit, 1)));
    }
}

#[test]
fn test_mm256_blendv_pd_all_from_a() {
    let solver = Solver::new();
    let (a, b) = pair64(&solver);
    let mask = ymm_reg("mask_a");
    assert_sign_bits(&mask, &solver, &[0, 0, 0, 0]);

    let output = mm256_blendv_pd(&a, &b, &mask, &solver);

    solver.assert(output.eq(a).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blendv_pd_all_from_b() {
    let solver = Solver::new();
    let (a, b) = pair64(&solver);
    let mask = ymm_reg("mask_b");
    assert_sign_bits(&mask, &solver, &[1, 1, 1, 1]);

    let output = mm256_blendv_pd(&a, &b, &mask, &solver);

    solver.assert(output.eq(b).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blendv_pd_alternating() {
    let solver = Solver::new();
    let (a, b) = pair64(&solver);
    let mask = ymm_reg("mask_alt");
    assert_sign_bits(&mask, &solver, &[0, 1, 0, 1]);

    let output = mm256_blendv_pd(&a, &b, &mask, &solver);
    let expected = construct_ymm_reg_from_elements(64, &[(&a, 0), (&b, 1), (&a, 2), (&b, 3)]);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_blendv_pd_symbolic_mask() {
    let solver = Solver::new();
    let (a, b) = pair64(&solver);
    let mask = ymm_reg("mask_symbolic");
    let output = mm256_blendv_pd(&a, &b, &mask, &solver);
    let expected = construct_ymm_reg_from_elements(64, &[(&b, 0), (&b, 1), (&a, 2), (&a, 3)]);

    solver.assert(output.eq(expected));
    assert_eq!(solver.check(), SatResult::Sat);
    let model = solver.get_model().expect("sat model");
    let expected_bits = [1, 1, 0, 0];
    for (element_idx, expected_bit) in expected_bits.iter().enumerate() {
        let high = element_idx as u32 * 64 + 63;
        let bit = model
            .eval(&mask.extract(high, high), true)
            .expect("mask sign bit")
            .as_u64()
            .expect("mask sign bit numeral");
        assert_eq!(bit, *expected_bit);
    }
}

#[test]
fn test_blendv_256_pd_canonicalization() {
    let solver = Solver::new();
    let a = ymm_reg("canon_a");
    let b = ymm_reg("canon_b");
    let mask = BV::new_const("canon_blendv_pd_mask", 256);

    let _result = mm256_blendv_pd(&a, &b, &mask, &solver);

    solver.assert(mask.extract(62, 0).eq(BV::from_u64(0, 63)).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}
