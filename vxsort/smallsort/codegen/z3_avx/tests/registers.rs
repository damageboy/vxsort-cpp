use z3::ast::BV;
use z3::{SatResult, Solver};
use z3_avx::registers::{ymm_reg, zmm_reg};

#[test]
fn registers_have_expected_widths() {
    assert_eq!(ymm_reg("ymm0").get_size(), 256);
    assert_eq!(zmm_reg("zmm0").get_size(), 512);
}

#[test]
fn construct_ymm_reg_from_elements_reorders_lanes() {
    let solver = Solver::new();
    let source = z3_avx::registers::ymm_reg_with_32b_values(
        "source",
        &solver,
        &[10, 11, 12, 13, 14, 15, 16, 17],
    );

    let reordered = z3_avx::registers::construct_ymm_reg_from_elements(
        32,
        &[
            (&source, 7),
            (&source, 6),
            (&source, 5),
            (&source, 4),
            (&source, 3),
            (&source, 2),
            (&source, 1),
            (&source, 0),
        ],
    );

    for lane in 0..8 {
        let low = lane * 32;
        let expected = BV::from_u64(17 - lane as u64, 32);

        solver.push();
        solver.assert(reordered.extract(low + 31, low).eq(expected).not());
        assert_eq!(solver.check(), SatResult::Unsat);
        solver.pop(1);
    }
}

#[test]
fn ymm_reg_with_unique_values_constrains_lanes_distinct() {
    let solver = Solver::new();
    let reg = z3_avx::registers::ymm_reg_with_unique_values("unique", &solver, 32);

    solver.assert(reg.extract(31, 0).eq(reg.extract(63, 32)));
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn ymm_reg_reversed_constrains_reverse_lane_order() {
    let solver = Solver::new();
    let input = z3_avx::registers::ymm_reg_with_32b_values(
        "input",
        &solver,
        &[10, 11, 12, 13, 14, 15, 16, 17],
    );
    let reversed = z3_avx::registers::ymm_reg_reversed("reversed", &solver, &input, 32);

    for lane in 0..8 {
        let low = lane * 32;
        let expected = BV::from_u64(17 - lane as u64, 32);

        solver.push();
        solver.assert(reversed.extract(low + 31, low).eq(expected).not());
        assert_eq!(solver.check(), SatResult::Unsat);
        solver.pop(1);
    }
}

#[test]
fn ymm_reg_with_32b_values_packs_lanes() {
    let solver = Solver::new();
    let reg = z3_avx::registers::ymm_reg_with_32b_values(
        "values",
        &solver,
        &[10, 11, 12, 13, 14, 15, 16, 17],
    );

    for lane in 0..8 {
        let low = lane * 32;
        let expected = BV::from_u64(10 + lane as u64, 32);

        solver.push();
        solver.assert(reg.extract(low + 31, low).eq(expected).not());
        assert_eq!(solver.check(), SatResult::Unsat);
        solver.pop(1);
    }
}
