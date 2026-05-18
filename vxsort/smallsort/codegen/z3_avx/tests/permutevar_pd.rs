use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::permutevar::{mm256_permutevar_pd, mm512_mask_permutevar_pd, mm512_permutevar_pd};
use z3_avx::registers::{
    construct_ymm_reg_from_elements, construct_zmm_reg_from_elements, ymm_reg,
    ymm_reg_with_unique_values, zmm_reg, zmm_reg_with_unique_values,
};

fn assert_unsat_if_nonzero(solver: &Solver, value: &BV) {
    solver.push();
    solver.assert(value.eq(BV::from_u64(0, value.get_size())).not());
    assert_eq!(solver.check(), SatResult::Unsat);
    solver.pop(1);
}

fn pd_ctrl(name: &str, solver: &Solver, total_bits: u32, controls: &[u64]) -> BV {
    assert_eq!(controls.len(), (total_bits / 64) as usize);

    let ctrl = BV::new_const(name, total_bits);
    for (element, control) in controls.iter().copied().enumerate() {
        let bit = element as u32 * 64 + 1;
        solver.assert(ctrl.extract(bit, bit).eq(BV::from_u64(control, 1)));
    }
    ctrl
}

#[test]
fn test_mm256_permutevar_pd_identity_permute() {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("a_identity_pd_256", &solver, 64);
    let ctrl = pd_ctrl("ctrl_identity_pd_256", &solver, 256, &[0, 1, 0, 1]);

    let output = mm256_permutevar_pd(&a, &ctrl, &solver);

    solver.assert(output.eq(a).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permutevar_pd_swap_within_lanes() {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("a_swap_pd_256", &solver, 64);
    let ctrl = pd_ctrl("ctrl_swap_pd_256", &solver, 256, &[1, 0, 1, 0]);

    let output = mm256_permutevar_pd(&a, &ctrl, &solver);
    let expected = construct_ymm_reg_from_elements(64, &[(&a, 1), (&a, 0), (&a, 3), (&a, 2)]);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permutevar_pd_broadcast_first_within_lanes() {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("a_broadcast_first_pd_256", &solver, 64);
    let ctrl = pd_ctrl("ctrl_broadcast_first_pd_256", &solver, 256, &[0, 0, 0, 0]);

    let output = mm256_permutevar_pd(&a, &ctrl, &solver);
    let expected = construct_ymm_reg_from_elements(64, &[(&a, 0), (&a, 0), (&a, 2), (&a, 2)]);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permutevar_pd_broadcast_second_within_lanes() {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("a_broadcast_second_pd_256", &solver, 64);
    let ctrl = pd_ctrl("ctrl_broadcast_second_pd_256", &solver, 256, &[1, 1, 1, 1]);

    let output = mm256_permutevar_pd(&a, &ctrl, &solver);
    let expected = construct_ymm_reg_from_elements(64, &[(&a, 1), (&a, 1), (&a, 3), (&a, 3)]);

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_permutevar_pd_identity_permute() {
    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("a_identity_pd_512", &solver, 64);
    let ctrl = pd_ctrl(
        "ctrl_identity_pd_512",
        &solver,
        512,
        &[0, 1, 0, 1, 0, 1, 0, 1],
    );

    let output = mm512_permutevar_pd(&a, &ctrl, &solver);

    solver.assert(output.eq(a).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_permutevar_pd_swap_within_lanes() {
    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("a_swap_pd_512", &solver, 64);
    let ctrl = pd_ctrl("ctrl_swap_pd_512", &solver, 512, &[1, 0, 1, 0, 1, 0, 1, 0]);

    let output = mm512_permutevar_pd(&a, &ctrl, &solver);
    let expected = construct_zmm_reg_from_elements(
        64,
        &[
            (&a, 1),
            (&a, 0),
            (&a, 3),
            (&a, 2),
            (&a, 5),
            (&a, 4),
            (&a, 7),
            (&a, 6),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_permutevar_pd_broadcast_first_within_lanes() {
    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("a_broadcast_first_pd_512", &solver, 64);
    let ctrl = pd_ctrl("ctrl_broadcast_first_pd_512", &solver, 512, &[0; 8]);

    let output = mm512_permutevar_pd(&a, &ctrl, &solver);
    let expected = construct_zmm_reg_from_elements(
        64,
        &[
            (&a, 0),
            (&a, 0),
            (&a, 2),
            (&a, 2),
            (&a, 4),
            (&a, 4),
            (&a, 6),
            (&a, 6),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_permutevar_pd_broadcast_second_within_lanes() {
    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("a_broadcast_second_pd_512", &solver, 64);
    let ctrl = pd_ctrl("ctrl_broadcast_second_pd_512", &solver, 512, &[1; 8]);

    let output = mm512_permutevar_pd(&a, &ctrl, &solver);
    let expected = construct_zmm_reg_from_elements(
        64,
        &[
            (&a, 1),
            (&a, 1),
            (&a, 3),
            (&a, 3),
            (&a, 5),
            (&a, 5),
            (&a, 7),
            (&a, 7),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutevar_pd_mask_all_zeros() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_zero_mask_pd", &solver, 64);
    let a = zmm_reg_with_unique_values("a_zero_mask_pd", &solver, 64);
    let ctrl = zmm_reg("ctrl_zero_mask_pd");
    let mask = BV::from_u64(0, 8);

    let output = mm512_mask_permutevar_pd(&src, &mask, &a, &ctrl, &solver);

    solver.assert(output.eq(src).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutevar_pd_identity_permute() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_identity_mask_pd", &solver, 64);
    let a = zmm_reg_with_unique_values("a_identity_mask_pd", &solver, 64);
    let ctrl = pd_ctrl(
        "ctrl_identity_mask_pd",
        &solver,
        512,
        &[0, 1, 0, 1, 0, 1, 0, 1],
    );
    let mask = BV::from_u64(0xFF, 8);

    let output = mm512_mask_permutevar_pd(&src, &mask, &a, &ctrl, &solver);

    solver.assert(output.eq(a).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutevar_pd_swap_within_lanes() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_swap_mask_pd", &solver, 64);
    let a = zmm_reg_with_unique_values("a_swap_mask_pd", &solver, 64);
    let ctrl = pd_ctrl("ctrl_swap_mask_pd", &solver, 512, &[1, 0, 1, 0, 1, 0, 1, 0]);
    let mask = BV::from_u64(0xFF, 8);

    let output = mm512_mask_permutevar_pd(&src, &mask, &a, &ctrl, &solver);
    let expected = construct_zmm_reg_from_elements(
        64,
        &[
            (&a, 1),
            (&a, 0),
            (&a, 3),
            (&a, 2),
            (&a, 5),
            (&a, 4),
            (&a, 7),
            (&a, 6),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutevar_pd_broadcast_within_lanes() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_broadcast_mask_pd", &solver, 64);
    let a = zmm_reg_with_unique_values("a_broadcast_mask_pd", &solver, 64);
    let ctrl = pd_ctrl("ctrl_broadcast_mask_pd", &solver, 512, &[0; 8]);
    let mask = BV::from_u64(0xFF, 8);

    let output = mm512_mask_permutevar_pd(&src, &mask, &a, &ctrl, &solver);
    let expected = construct_zmm_reg_from_elements(
        64,
        &[
            (&a, 0),
            (&a, 0),
            (&a, 2),
            (&a, 2),
            (&a, 4),
            (&a, 4),
            (&a, 6),
            (&a, 6),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permutevar_pd_canonicalizes_control_dead_bits() {
    let solver = Solver::new();
    let a = ymm_reg("a_canonical_pd_256");
    let ctrl = BV::new_const("ctrl_canonical_pd_256", 256);

    let _output = mm256_permutevar_pd(&a, &ctrl, &solver);

    for element in 0..4 {
        let base = element * 64;
        assert_unsat_if_nonzero(&solver, &ctrl.extract(base, base));
        assert_unsat_if_nonzero(&solver, &ctrl.extract(base + 63, base + 2));
    }
}

#[test]
fn test_mm512_permutevar_pd_canonicalizes_control_dead_bits() {
    let solver = Solver::new();
    let a = zmm_reg("a_canonical_pd_512");
    let ctrl = BV::new_const("ctrl_canonical_pd_512", 512);

    let _output = mm512_permutevar_pd(&a, &ctrl, &solver);

    for element in 0..8 {
        let base = element * 64;
        assert_unsat_if_nonzero(&solver, &ctrl.extract(base, base));
        assert_unsat_if_nonzero(&solver, &ctrl.extract(base + 63, base + 2));
    }
}

#[test]
fn test_mm512_mask_permutevar_pd_canonicalizes_dead_mask_bits() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_wide_mask_pd", &solver, 64);
    let a = zmm_reg_with_unique_values("a_wide_mask_pd", &solver, 64);
    let ctrl = pd_ctrl("ctrl_wide_mask_pd", &solver, 512, &[0, 1, 0, 1, 0, 1, 0, 1]);
    let mask = BV::new_const("k_wide_pd", 64);

    let _output = mm512_mask_permutevar_pd(&src, &mask, &a, &ctrl, &solver);

    assert_unsat_if_nonzero(&solver, &mask.extract(63, 8));
}
