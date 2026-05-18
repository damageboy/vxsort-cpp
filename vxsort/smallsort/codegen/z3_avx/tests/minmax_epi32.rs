use z3::ast::{BV, Bool};
use z3::{SatResult, Solver};
use z3_avx::lanes::concat_msb_first;
use z3_avx::minmax::{mm256_max_epi32, mm256_min_epi32, mm512_max_epi32, mm512_min_epi32};
use z3_avx::registers::{ymm_reg, ymm_reg_with_32b_values, zmm_reg_with_32b_values};

fn packed_32(values: &[u64], total_bits: u32) -> BV {
    assert_eq!(values.len(), (total_bits / 32) as usize);
    let elements: Vec<BV> = values
        .iter()
        .map(|value| BV::from_u64(*value, 32))
        .collect();
    let reversed: Vec<BV> = elements.into_iter().rev().collect();
    concat_msb_first(&reversed)
}

#[test]
fn test_mm256_min_epi32_concrete() {
    let solver = Solver::new();
    let a = ymm_reg_with_32b_values("a", &solver, &[5, 3, 7, 1, 8, 2, 6, 4]);
    let b = ymm_reg_with_32b_values("b", &solver, &[2, 8, 4, 6, 1, 5, 3, 7]);
    let expected = packed_32(&[2, 3, 4, 1, 1, 2, 3, 4], 256);

    let result = mm256_min_epi32(&a, &b);

    solver.assert(result.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_max_epi32_concrete() {
    let solver = Solver::new();
    let a = ymm_reg_with_32b_values("a", &solver, &[5, 3, 7, 1, 8, 2, 6, 4]);
    let b = ymm_reg_with_32b_values("b", &solver, &[2, 8, 4, 6, 1, 5, 3, 7]);
    let expected = packed_32(&[5, 8, 7, 6, 8, 5, 6, 7], 256);

    let result = mm256_max_epi32(&a, &b);

    solver.assert(result.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_min_epi32_signed() {
    let solver = Solver::new();
    let neg_one = (1_u64 << 32) - 1;
    let a = ymm_reg_with_32b_values("a", &solver, &[neg_one; 8]);
    let b = ymm_reg_with_32b_values("b", &solver, &[1; 8]);
    let expected = packed_32(&[neg_one; 8], 256);

    let result = mm256_min_epi32(&a, &b);

    solver.assert(result.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm256_minmax_epi32_preserves_elements() {
    let solver = Solver::new();
    let a = ymm_reg("a");
    let b = ymm_reg("b");
    let lo = mm256_min_epi32(&a, &b);
    let hi = mm256_max_epi32(&a, &b);

    for lane in 0..8 {
        let low = lane * 32;
        let high = low + 31;
        let a_elem = a.extract(high, low);
        let b_elem = b.extract(high, low);
        let lo_elem = lo.extract(high, low);
        let hi_elem = hi.extract(high, low);
        let keeps_order = Bool::and(&[lo_elem.eq(&a_elem), hi_elem.eq(&b_elem)]);
        let swaps_order = Bool::and(&[lo_elem.eq(&b_elem), hi_elem.eq(&a_elem)]);

        solver.assert(Bool::or(&[keeps_order, swaps_order]).not());
    }

    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_min_epi32_concrete() {
    let solver = Solver::new();
    let a = zmm_reg_with_32b_values(
        "a",
        &solver,
        &[5, 3, 7, 1, 8, 2, 6, 4, 10, 12, 9, 11, 16, 14, 13, 15],
    );
    let b = zmm_reg_with_32b_values(
        "b",
        &solver,
        &[2, 8, 4, 6, 1, 5, 3, 7, 11, 9, 12, 10, 13, 15, 16, 14],
    );
    let expected = packed_32(&[2, 3, 4, 1, 1, 2, 3, 4, 10, 9, 9, 10, 13, 14, 13, 14], 512);

    let result = mm512_min_epi32(&a, &b);

    solver.assert(result.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}

#[test]
fn test_mm512_max_epi32_concrete() {
    let solver = Solver::new();
    let a = zmm_reg_with_32b_values(
        "a",
        &solver,
        &[5, 3, 7, 1, 8, 2, 6, 4, 10, 12, 9, 11, 16, 14, 13, 15],
    );
    let b = zmm_reg_with_32b_values(
        "b",
        &solver,
        &[2, 8, 4, 6, 1, 5, 3, 7, 11, 9, 12, 10, 13, 15, 16, 14],
    );
    let expected = packed_32(
        &[5, 8, 7, 6, 8, 5, 6, 7, 11, 12, 12, 11, 16, 15, 16, 15],
        512,
    );

    let result = mm512_max_epi32(&a, &b);

    solver.assert(result.eq(expected).not());
    assert_eq!(solver.check(), SatResult::Unsat);
}
