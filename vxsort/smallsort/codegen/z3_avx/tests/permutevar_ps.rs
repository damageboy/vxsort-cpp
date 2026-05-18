use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::permutevar::{mm256_permutevar_ps, mm512_mask_permutevar_ps, mm512_permutevar_ps};
use z3_avx::registers::{
    construct_ymm_reg_from_elements, construct_zmm_reg_from_elements, ymm_reg,
    ymm_reg_with_32b_values, ymm_reg_with_unique_values, zmm_reg, zmm_reg_with_32b_values,
    zmm_reg_with_unique_values,
};

fn assert_unsat_if_nonzero(solver: &Solver, value: &BV) {
    solver.push();
    solver.assert(value.eq(BV::from_u64(0, value.get_size())).not());
    assert_eq!(solver.check(), SatResult::Unsat);
    solver.pop(1);
}

#[test]
fn test_mm256_permutevar_ps_identity_permute() {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("a_identity_256", &solver, 32);
    let ctrl = ymm_reg_with_32b_values("ctrl_identity_256", &solver, &[0, 1, 2, 3, 0, 1, 2, 3]);

    let output = mm256_permutevar_ps(&a, &ctrl, &solver);

    solver.assert(output.eq(a).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permutevar_ps_reverse_within_lanes() {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("a_reverse_256", &solver, 32);
    let ctrl = ymm_reg_with_32b_values("ctrl_reverse_256", &solver, &[3, 2, 1, 0, 3, 2, 1, 0]);

    let output = mm256_permutevar_ps(&a, &ctrl, &solver);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&a, 3),
            (&a, 2),
            (&a, 1),
            (&a, 0),
            (&a, 7),
            (&a, 6),
            (&a, 5),
            (&a, 4),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permutevar_ps_broadcast_within_lanes() {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("a_broadcast_256", &solver, 32);
    let ctrl = ymm_reg_with_32b_values("ctrl_broadcast_256", &solver, &[0; 8]);

    let output = mm256_permutevar_ps(&a, &ctrl, &solver);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&a, 0),
            (&a, 0),
            (&a, 0),
            (&a, 0),
            (&a, 4),
            (&a, 4),
            (&a, 4),
            (&a, 4),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permutevar_ps_mixed_permute() {
    let solver = Solver::new();
    let a = ymm_reg_with_unique_values("a_mixed_256", &solver, 32);
    let ctrl = ymm_reg_with_32b_values("ctrl_mixed_256", &solver, &[1, 0, 3, 2, 2, 3, 0, 1]);

    let output = mm256_permutevar_ps(&a, &ctrl, &solver);
    let expected = construct_ymm_reg_from_elements(
        32,
        &[
            (&a, 1),
            (&a, 0),
            (&a, 3),
            (&a, 2),
            (&a, 6),
            (&a, 7),
            (&a, 4),
            (&a, 5),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_permutevar_ps_identity_permute() {
    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("a_identity_512", &solver, 32);
    let ctrl = zmm_reg_with_32b_values(
        "ctrl_identity_512",
        &solver,
        &[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    );

    let output = mm512_permutevar_ps(&a, &ctrl, &solver);

    solver.assert(output.eq(a).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_permutevar_ps_reverse_within_lanes() {
    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("a_reverse_512", &solver, 32);
    let ctrl = zmm_reg_with_32b_values(
        "ctrl_reverse_512",
        &solver,
        &[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
    );

    let output = mm512_permutevar_ps(&a, &ctrl, &solver);
    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&a, 3),
            (&a, 2),
            (&a, 1),
            (&a, 0),
            (&a, 7),
            (&a, 6),
            (&a, 5),
            (&a, 4),
            (&a, 11),
            (&a, 10),
            (&a, 9),
            (&a, 8),
            (&a, 15),
            (&a, 14),
            (&a, 13),
            (&a, 12),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_permutevar_ps_broadcast_within_lanes() {
    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("a_broadcast_512", &solver, 32);
    let ctrl = zmm_reg_with_32b_values("ctrl_broadcast_512", &solver, &[3; 16]);

    let output = mm512_permutevar_ps(&a, &ctrl, &solver);
    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&a, 3),
            (&a, 3),
            (&a, 3),
            (&a, 3),
            (&a, 7),
            (&a, 7),
            (&a, 7),
            (&a, 7),
            (&a, 11),
            (&a, 11),
            (&a, 11),
            (&a, 11),
            (&a, 15),
            (&a, 15),
            (&a, 15),
            (&a, 15),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_permutevar_ps_alternating_pattern() {
    let solver = Solver::new();
    let a = zmm_reg_with_unique_values("a_alternating_512", &solver, 32);
    let ctrl = zmm_reg_with_32b_values(
        "ctrl_alternating_512",
        &solver,
        &[0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2],
    );

    let output = mm512_permutevar_ps(&a, &ctrl, &solver);
    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&a, 0),
            (&a, 2),
            (&a, 0),
            (&a, 2),
            (&a, 4),
            (&a, 6),
            (&a, 4),
            (&a, 6),
            (&a, 8),
            (&a, 10),
            (&a, 8),
            (&a, 10),
            (&a, 12),
            (&a, 14),
            (&a, 12),
            (&a, 14),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutevar_ps_mask_all_zeros() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_zero_mask_ps", &solver, 32);
    let a = zmm_reg_with_unique_values("a_zero_mask_ps", &solver, 32);
    let ctrl = zmm_reg_with_32b_values(
        "ctrl_zero_mask_ps",
        &solver,
        &[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    );
    let mask = BV::from_u64(0, 16);

    let output = mm512_mask_permutevar_ps(&src, &mask, &a, &ctrl, &solver);

    solver.assert(output.eq(src).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutevar_ps_identity_permute() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_identity_mask_ps", &solver, 32);
    let a = zmm_reg_with_unique_values("a_identity_mask_ps", &solver, 32);
    let ctrl = zmm_reg_with_32b_values(
        "ctrl_identity_mask_ps",
        &solver,
        &[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    );
    let mask = BV::from_u64(0xFFFF, 16);

    let output = mm512_mask_permutevar_ps(&src, &mask, &a, &ctrl, &solver);

    solver.assert(output.eq(a).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutevar_ps_reverse_within_lanes() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_reverse_mask_ps", &solver, 32);
    let a = zmm_reg_with_unique_values("a_reverse_mask_ps", &solver, 32);
    let ctrl = zmm_reg_with_32b_values(
        "ctrl_reverse_mask_ps",
        &solver,
        &[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0],
    );
    let mask = BV::from_u64(0xFFFF, 16);

    let output = mm512_mask_permutevar_ps(&src, &mask, &a, &ctrl, &solver);
    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&a, 3),
            (&a, 2),
            (&a, 1),
            (&a, 0),
            (&a, 7),
            (&a, 6),
            (&a, 5),
            (&a, 4),
            (&a, 11),
            (&a, 10),
            (&a, 9),
            (&a, 8),
            (&a, 15),
            (&a, 14),
            (&a, 13),
            (&a, 12),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_mask_permutevar_ps_broadcast_within_lanes() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_broadcast_mask_ps", &solver, 32);
    let a = zmm_reg_with_unique_values("a_broadcast_mask_ps", &solver, 32);
    let ctrl = zmm_reg_with_32b_values("ctrl_broadcast_mask_ps", &solver, &[0; 16]);
    let mask = BV::from_u64(0xFFFF, 16);

    let output = mm512_mask_permutevar_ps(&src, &mask, &a, &ctrl, &solver);
    let expected = construct_zmm_reg_from_elements(
        32,
        &[
            (&a, 0),
            (&a, 0),
            (&a, 0),
            (&a, 0),
            (&a, 4),
            (&a, 4),
            (&a, 4),
            (&a, 4),
            (&a, 8),
            (&a, 8),
            (&a, 8),
            (&a, 8),
            (&a, 12),
            (&a, 12),
            (&a, 12),
            (&a, 12),
        ],
    );

    solver.assert(output.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_permutevar_ps_canonicalizes_control_high_bits() {
    let solver = Solver::new();
    let a = ymm_reg("a_canonical_ps_256");
    let ctrl = BV::new_const("ctrl_canonical_ps_256", 256);

    let _output = mm256_permutevar_ps(&a, &ctrl, &solver);

    for element in 0..8 {
        let low = element * 32 + 2;
        let high = element * 32 + 31;
        assert_unsat_if_nonzero(&solver, &ctrl.extract(high, low));
    }
}

#[test]
fn test_mm512_permutevar_ps_canonicalizes_control_high_bits() {
    let solver = Solver::new();
    let a = zmm_reg("a_canonical_ps_512");
    let ctrl = BV::new_const("ctrl_canonical_ps_512", 512);

    let _output = mm512_permutevar_ps(&a, &ctrl, &solver);

    for element in 0..16 {
        let low = element * 32 + 2;
        let high = element * 32 + 31;
        assert_unsat_if_nonzero(&solver, &ctrl.extract(high, low));
    }
}

#[test]
fn test_mm512_mask_permutevar_ps_canonicalizes_dead_mask_bits() {
    let solver = Solver::new();
    let src = zmm_reg_with_unique_values("src_wide_mask_ps", &solver, 32);
    let a = zmm_reg_with_unique_values("a_wide_mask_ps", &solver, 32);
    let ctrl = zmm_reg_with_32b_values(
        "ctrl_wide_mask_ps",
        &solver,
        &[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    );
    let mask = BV::new_const("k_wide_ps", 64);

    let _output = mm512_mask_permutevar_ps(&src, &mask, &a, &ctrl, &solver);

    assert_unsat_if_nonzero(&solver, &mask.extract(63, 16));
}
